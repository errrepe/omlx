// SPDX-License-Identifier: Apache-2.0
//
// expert_bank_wrap: wrap page-aligned mmap'd full-bank expert tensors as
// mx.arrays via newBufferWithBytesNoCopy (zero Metal allocation), feeding the
// STOCK mlx::core::gather_qmm path unchanged.
//
// Adapted from jundot/omlx PR #3437 (qwen4_moe_stream) by @alytaphoenix,
// Apache-2.0. Differences from the source (documented in
// docs/expert-streaming.md, Fase M): page size is manifest-level data passed
// at mmap time (no hardcoded 16 KiB assumption in consumers); the artifact is
// produced by tools/repack_fullbank.py with a content fingerprint, and the
// loader (omlx/patches/expert_streaming/fullbank.py) refuses stale/corrupt
// artifacts before any tensor is served.
//
// No custom Metal kernels. make_buffer wrappers MUST be released with
// allocator::release (mlx/allocator.h) -- a no-op deleter would leak one
// wrapper handle per wrapped tensor per load/unload cycle. release() frees
// only the lightweight MTL::Buffer wrapper object, never the mmap'd bytes
// (the mapping owns those).

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/allocator.h"
#include "mlx/array.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace omlx_ebw {

namespace mx = mlx::core;

// Minimum page alignment required for newBufferWithBytesNoCopy. The artifact
// producer pads every tensor to the manifest page size (>= this on every
// Apple Silicon platform, 16 KiB). mmap_artifact carries the concrete page
// size so wrap_tensor can validate offsets against it.
static constexpr size_t kMinPage = 16384;

struct Mapping {
  void* base = nullptr;
  size_t size = 0;
  int fd = -1;
  size_t page = kMinPage;  // manifest page size (power of two, >= kMinPage)
  long refcount = 0;       // outstanding wrapped arrays on this mapping
  bool closed = false;     // close_artifact ran; unmap deferred to refcount 0
};

static std::mutex g_mu;
static std::unordered_map<int, Mapping> g_maps;
static int g_next_id = 1;

static void unmap_locked(std::unordered_map<int, Mapping>::iterator it) {
  ::munmap(it->second.base, it->second.size);
  ::close(it->second.fd);
  g_maps.erase(it);
}

// Called by each wrapped array's deleter: drop one ref and, if close_artifact
// already ran and this was the last live array, unmap now. (Ported from PR
// #3437 "refcounted deferred munmap": a GPU touch of a wrapped array after
// close() can never fault on unmapped memory.)
static void release_ref(int id) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_maps.find(id);
  if (it == g_maps.end()) return;
  if (it->second.refcount > 0) it->second.refcount--;
  if (it->second.closed && it->second.refcount == 0) unmap_locked(it);
}

static size_t align_up_to(size_t n, size_t a) { return (n + a - 1) / a * a; }

static mx::Dtype dtype_from_str(const std::string& s) {
  if (s == "uint32" || s == "U32") return mx::uint32;
  if (s == "bfloat16" || s == "BF16") return mx::bfloat16;
  if (s == "float16" || s == "F16") return mx::float16;
  if (s == "uint16" || s == "U16") return mx::uint16;
  if (s == "uint8" || s == "U8") return mx::uint8;
  throw std::runtime_error("wrap_tensor: unsupported dtype: " + s);
}

static constexpr size_t kMaxTensorBytes = 1ull << 40;  // 1 TiB sanity cap

}  // namespace omlx_ebw

NB_MODULE(_ext, m) {
  m.doc() = "Full-bank external wrap for MoE expert streaming (mmap'd page-aligned artifact)";

  using namespace omlx_ebw;

  // ABI canary (same pattern as glm_moe_dsa / qwen35_prefill / PR #3437): if
  // the nanobind ABI does not match the mlx wheel, passing an mx.array here
  // fails and fast.py disables the native path instead of crashing.
  m.def(
      "abi_probe",
      [](const mx::array& a) { return static_cast<int64_t>(a.size()); },
      "a"_a);

  // mmap the artifact read-only/shared; returns an opaque id. page is the
  // manifest page size (power of two, >= 16 KiB): wrap_tensor enforces
  // offsets are aligned to it so the pointer handed to
  // newBufferWithBytesNoCopy is page-aligned.
  m.def(
      "mmap_artifact",
      [](const std::string& path, size_t page) -> int {
        if (page < kMinPage || (page & (page - 1)) != 0)
          throw std::runtime_error("mmap_artifact: invalid page size: " +
                                   std::to_string(page));
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("mmap_artifact: open failed: " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
          ::close(fd);
          throw std::runtime_error("mmap_artifact: fstat failed: " + path);
        }
        void* base = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                            MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
          ::close(fd);
          throw std::runtime_error("mmap_artifact: mmap failed: " + path);
        }
        std::lock_guard<std::mutex> lk(g_mu);
        int id = g_next_id++;
        g_maps[id] = Mapping{base, static_cast<size_t>(st.st_size), fd, page, 0, false};
        return id;
      },
      "path"_a, "page"_a);

  // Mark the mapping closed. Unmap immediately only if no wrapped arrays
  // still point into it; otherwise defer the munmap to the last deleter.
  m.def(
      "close_artifact",
      [](int id) {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_maps.find(id);
        if (it == g_maps.end()) return;
        it->second.closed = true;
        if (it->second.refcount == 0) unmap_locked(it);
      },
      "id"_a);

  // Total bytes currently mmap'd across all live artifacts. The Python side
  // feeds this to the external-wired provider registry (worst-case figure:
  // all mapped artifact pages; observability-only in health, never charged
  // against a ceiling -- see omlx/utils/proc_memory.py).
  m.def("mapped_bytes", []() -> size_t {
    std::lock_guard<std::mutex> lk(g_mu);
    size_t total = 0;
    for (auto& kv : g_maps) total += kv.second.size;
    return total;
  });

  // Return an mx.array viewing [offset, offset+length) of artifact `id` as
  // `shape`/`dtype`, backed by external mmap memory (no copy). offset MUST be
  // page-aligned (guaranteed by the repack tool) so the pointer is aligned
  // for newBufferWithBytesNoCopy. The MTLBuffer is created over a page-
  // rounded length (each tensor's region is page-padded in the artifact, so
  // the rounded span stays inside the file and never overlaps the next
  // tensor); the array itself only addresses the logical `length` bytes.
  m.def(
      "wrap_tensor",
      [](int id, size_t offset, size_t length, std::vector<int> shape,
         const std::string& dtype) -> mx::array {
        void* base = nullptr;
        size_t page = kMinPage;
        size_t buf_len = 0;
        {
          std::lock_guard<std::mutex> lk(g_mu);
          auto it = g_maps.find(id);
          if (it == g_maps.end())
            throw std::runtime_error("wrap_tensor: unknown artifact id");
          if (it->second.closed)
            throw std::runtime_error("wrap_tensor: artifact already closed");
          page = it->second.page;
          if (offset % page != 0)
            throw std::runtime_error("wrap_tensor: offset not page-aligned");
          if (length == 0 || length > kMaxTensorBytes)
            throw std::runtime_error("wrap_tensor: invalid length");
          buf_len = align_up_to(length, page);
          if (offset + buf_len > it->second.size)
            throw std::runtime_error("wrap_tensor: region out of bounds");
          base = it->second.base;
          it->second.refcount++;  // one ref per wrapped array; deleter drops it
        }
        mx::allocator::Buffer buf =
            mx::allocator::make_buffer(static_cast<char*>(base) + offset, buf_len);
        mx::Shape sh(shape.begin(), shape.end());
        // Sanctioned single-step constructor (same as PR #3437):
        // array(Buffer, Shape, Dtype, Deleter) marks the array available; no
        // set_status needed. The deleter releases the make_buffer wrapper AND
        // drops this mapping's refcount (deferred-unmap safe).
        return mx::array(buf, std::move(sh), dtype_from_str(dtype),
                        [id](mx::allocator::Buffer b) {
                          mx::allocator::release(b);
                          release_ref(id);
                        });
      },
      "id"_a, "offset"_a, "length"_a, "shape"_a, "dtype"_a);
}
