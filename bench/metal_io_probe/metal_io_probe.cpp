// Final cold A/B: arms INTERLEAVED within each rep (no time-drift bias),
// cursor starts at shard `start_file` and never reuses a region.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
static size_t full_read(int fd, size_t off, uint8_t* dst, size_t n) {
  size_t done = 0;
  while (done < n) {
    ssize_t g = pread(fd, dst + done, n - done, (off_t)(off + done));
    if (g <= 0) break;
    done += (size_t)g;
  }
  return done;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: micro5 <modeldir> [N] [gap] [reps] [start_file]\n"); return 2; }
  std::string dir = argv[1];
  int N = argc > 2 ? atoi(argv[2]) : 24;
  size_t GAPSL = argc > 3 ? strtoull(argv[3], nullptr, 10) : 32;
  int REPS = argc > 4 ? atoi(argv[4]) : 5;
  size_t START = argc > 5 ? strtoull(argv[5], nullptr, 10) : 8;

  std::vector<std::string> files;
  std::vector<size_t> sizes;
  DIR* d = opendir(dir.c_str());
  struct dirent* e;
  std::vector<std::string> got;
  while ((e = readdir(d))) {
    std::string n = e->d_name;
    if (n.size() >= 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) got.push_back(n);
  }
  closedir(d);
  std::sort(got.begin(), got.end());
  for (auto& g : got) {
    std::string p = dir + "/" + g;
    struct stat st;
    stat(p.c_str(), &st);
    files.push_back(p);
    sizes.push_back(st.st_size);
  }

  const size_t SLICE = 819200;
  const size_t GAP = GAPSL * SLICE;
  const size_t SPAN = (size_t)N * GAP;
  const size_t TOTAL = (size_t)N * SLICE;

  MTL::Device* dev = MTL::CreateSystemDefaultDevice();
  NS::Error* err = nullptr;
  MTL::IOCommandQueueDescriptor* d0 = MTL::IOCommandQueueDescriptor::alloc()->init();
  d0->setType(MTL::IOCommandQueueTypeConcurrent);
  d0->setMaxCommandBufferCount(64);
  d0->setMaxCommandsInFlight(64);
  d0->setPriority(MTL::IOPriorityHigh);
  MTL::IOCommandQueue* q = dev->newIOCommandQueue(d0, &err);

  struct Arm { const char* name; int kind; int depth; };
  std::vector<Arm> arms = {
      {"pread 1t",        0, 1},
      {"pread 1t + copy", 1, 1},
      {"pread 8t + copy", 2, 8},
      {"pread 16t + copy",2, 16},
      {"pread 24t + copy",2, 24},
      {"mtlio qd1",       3, 1},
      {"mtlio qd4",       3, 4},
      {"mtlio qd8",       3, 8},
      {"mtlio qd16",      3, 16},
      {"mtlio qd24",      3, 24},
  };

  size_t fi = START, off = 1ull << 30;
  bool pool_ok = true, any_bad = false;
  std::vector<std::vector<double>> cold(arms.size()), warm(arms.size());
  std::vector<bool> vok(arms.size(), true);

  printf("N=%d gap=%.1fMiB span=%.0fMiB payload=%.1fMiB reps=%d start_file=%zu files=%zu\n\n",
         N, GAP / 1048576.0, SPAN / 1048576.0, TOTAL / 1048576.0, REPS, START, files.size());

  for (int rep = 0; rep < REPS && pool_ok; rep++) {
    for (size_t ai = 0; ai < arms.size() && pool_ok; ai++) {
      auto& a = arms[ai];
      // fresh region
      while (true) {
        if (fi >= files.size()) { pool_ok = false; break; }
        if (off + SPAN + (64ull << 20) < sizes[fi]) break;
        fi++; off = 1ull << 30;
      }
      if (!pool_ok) break;
      size_t base = off;
      off += SPAN + (64ull << 20);

      const std::string& path = files[fi];
      MTL::IOFileHandle* fh = dev->newIOHandle(
          NS::URL::fileURLWithPath(NS::String::string(path.c_str(), NS::UTF8StringEncoding)), &err);
      int fd = open(path.c_str(), O_RDONLY);
      if (!fh || fd < 0) { fprintf(stderr, "open fail\n"); return 1; }

      std::vector<size_t> offs;
      for (int i = 0; i < N; i++) offs.push_back(base + (size_t)i * GAP);
      std::vector<uint8_t*> host(N), dst(N);
      std::vector<MTL::Buffer*> mb(N);
      for (int i = 0; i < N; i++) {
        host[i] = (uint8_t*)malloc(SLICE);
        dst[i] = (uint8_t*)malloc(SLICE);
        mb[i] = dev->newBuffer(SLICE, MTL::ResourceStorageModeShared);
        if (!host[i] || !dst[i] || !mb[i]) { fprintf(stderr, "alloc fail\n"); return 1; }
      }

      bool ok = true;
      auto run = [&]() -> double {
        auto t0 = clk::now();
        if (a.kind == 0) {
          for (int i = 0; i < N; i++) if (full_read(fd, offs[i], host[i], SLICE) != SLICE) ok = false;
        } else if (a.kind == 1) {
          for (int i = 0; i < N; i++) {
            if (full_read(fd, offs[i], host[i], SLICE) != SLICE) ok = false;
            memcpy(dst[i], host[i], SLICE);
          }
        } else if (a.kind == 2) {
          std::vector<std::thread> th;
          int live = 0;
          for (int i = 0; i < N; i++) {
            th.emplace_back([&, i] {
              if (full_read(fd, offs[i], host[i], SLICE) != SLICE) ok = false;
              memcpy(dst[i], host[i], SLICE);
            });
            if (++live >= a.depth) { for (auto& t : th) t.join(); th.clear(); live = 0; }
          }
          for (auto& t : th) t.join();
        } else {
          int per = (N + a.depth - 1) / a.depth;
          std::vector<MTL::IOCommandBuffer*> cbs;
          for (int dd = 0; dd < a.depth; dd++) {
            auto* cb = q->commandBuffer();
            for (int k = 0; k < per; k++) {
              int i = dd * per + k;
              if (i >= N) break;
              cb->loadBuffer(mb[i], 0, SLICE, fh, offs[i]);
            }
            cb->commit();
            cbs.push_back(cb);
          }
          for (auto* cb : cbs) {
            cb->waitUntilCompleted();
            if (cb->status() != MTL::IOStatusComplete) ok = false;
          }
        }
        return ms(t0, clk::now());
      };

      cold[ai].push_back(run());
      warm[ai].push_back(run());

      uint8_t* ref = (uint8_t*)malloc(SLICE);
      for (int i = 0; i < N; i++) {
        if (full_read(fd, offs[i], ref, SLICE) != SLICE) { ok = false; break; }
        const uint8_t* gotp = (a.kind == 3) ? (const uint8_t*)mb[i]->contents()
                                            : ((a.kind == 0) ? host[i] : dst[i]);
        if (memcmp(ref, gotp, SLICE) != 0) { ok = false; break; }
      }
      free(ref);
      if (!ok) { vok[ai] = false; any_bad = true; }
      for (int i = 0; i < N; i++) { free(host[i]); free(dst[i]); mb[i]->release(); }
      fh->release();
      close(fd);
    }
  }

  double mib = TOTAL / 1048576.0;
  printf("%-18s %8s %7s %8s %7s  %-6s  %s\n", "arm", "cold(ms)", "GB/s", "warm(ms)", "GB/s", "verify", "cold per-rep");
  for (size_t ai = 0; ai < arms.size(); ai++) {
    if (cold[ai].empty()) { printf("%-18s  NO DATA\n", arms[ai].name); continue; }
    double cm = 0, wm = 0;
    for (double v : cold[ai]) cm += v;
    for (double v : warm[ai]) wm += v;
    cm /= cold[ai].size(); wm /= warm[ai].size();
    // median of cold, more robust than mean here
    std::vector<double> s = cold[ai];
    std::sort(s.begin(), s.end());
    double med = s[s.size() / 2];
    printf("%-18s %8.2f %7.2f %8.2f %7.2f  %-6s  med=%.2f  [", arms[ai].name, cm,
           mib / (cm / 1000) / 1024, wm, mib / (wm / 1000) / 1024, vok[ai] ? "OK" : "BAD", med);
    for (double v : cold[ai]) printf("%.2f ", v);
    printf("]\n");
  }
  printf("\npool_exhausted=%s any_bad=%s\n", pool_ok ? "no" : "YES", any_bad ? "YES" : "no");
  return 0;
}
