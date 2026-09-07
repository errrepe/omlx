// Feasibility spike: does MTLIOCommandQueue work on this machine?
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : nullptr;
  if (!path) {
    fprintf(stderr, "usage: spike <file> [offset] [size]\n");
    return 2;
  }
  size_t offset = argc > 2 ? strtoull(argv[2], nullptr, 10) : 4096;
  size_t size = argc > 3 ? strtoull(argv[3], nullptr, 10) : (1u << 22);

  MTL::Device* dev = MTL::CreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
  printf("device: %s\n", dev->name()->utf8String());
  printf("supports family metal3: %d\n",
         dev->supportsFamily(MTL::GPUFamilyMetal3));

  NS::Error* err = nullptr;

  MTL::IOCommandQueueDescriptor* desc =
      MTL::IOCommandQueueDescriptor::alloc()->init();
  desc->setType(MTL::IOCommandQueueTypeConcurrent);
  desc->setMaxCommandBufferCount(16);
  desc->setMaxCommandsInFlight(16);
  desc->setPriority(MTL::IOPriorityHigh);

  MTL::IOCommandQueue* q = dev->newIOCommandQueue(desc, &err);
  if (!q) {
    fprintf(stderr, "newIOCommandQueue failed: %s\n",
            err ? err->localizedDescription()->utf8String() : "?");
    return 1;
  }
  printf("io queue: ok\n");

  NS::String* sp = NS::String::string(path, NS::UTF8StringEncoding);
  NS::URL* url = NS::URL::fileURLWithPath(sp);
  MTL::IOFileHandle* fh = dev->newIOHandle(url, &err);
  if (!fh) {
    fprintf(stderr, "newIOHandle failed: %s\n",
            err ? err->localizedDescription()->utf8String() : "?");
    return 1;
  }
  printf("io handle: ok\n");

  MTL::Buffer* buf = dev->newBuffer(size, MTL::ResourceStorageModeShared);
  if (!buf) { fprintf(stderr, "newBuffer failed\n"); return 1; }
  printf("buffer: %llu bytes\n", (unsigned long long)buf->length());

  // --- Metal IO read ---
  auto t0 = clk::now();
  MTL::IOCommandBuffer* cb = q->commandBuffer();
  cb->loadBuffer(buf, 0, size, fh, offset);
  cb->commit();
  cb->waitUntilCompleted();
  auto t1 = clk::now();
  MTL::IOStatus st = cb->status();
  NS::Error* cerr = cb->error();
  printf("io read: %.3f ms  status=%d  error=%s\n", ms(t0, t1), (int)st,
         cerr ? cerr->localizedDescription()->utf8String() : "none");

  // --- pread reference ---
  uint8_t* ref = (uint8_t*)malloc(size);
  int fd = open(path, O_RDONLY);
  auto t2 = clk::now();
  ssize_t got = pread(fd, ref, size, (off_t)offset);
  auto t3 = clk::now();
  printf("pread:   %.3f ms  got=%zd\n", ms(t2, t3), got);

  // --- compare ---
  int diff = 0;
  size_t first_diff = SIZE_MAX;
  const uint8_t* gotp = (const uint8_t*)buf->contents();
  for (size_t i = 0; i < size; i++) {
    if (gotp[i] != ref[i]) { if (!diff) first_diff = i; diff++; }
  }
  printf("bytes differing: %d (first at %llu)\n", diff,
         (unsigned long long)first_diff);

  // --- second (warm) IO read: does it hit the page cache? ---
  auto t4 = clk::now();
  MTL::IOCommandBuffer* cb2 = q->commandBuffer();
  cb2->loadBuffer(buf, 0, size, fh, offset);
  cb2->commit();
  cb2->waitUntilCompleted();
  auto t5 = clk::now();
  printf("io read (2nd): %.3f ms status=%d\n", ms(t4, t5), (int)cb2->status());

  // --- unaligned / small size sanity ---
  for (size_t sz : {size_t(64), size_t(1000), size_t(4097)}) {
    MTL::Buffer* tb = dev->newBuffer(sz, MTL::ResourceStorageModeShared);
    MTL::IOCommandBuffer* cbt = q->commandBuffer();
    cbt->loadBuffer(tb, 0, sz, fh, offset + 3);  // unaligned src offset
    cbt->commit();
    cbt->waitUntilCompleted();
    int d = 0;
    for (size_t i = 0; i < sz; i++)
      if (((const uint8_t*)tb->contents())[i] != ref[i + 3]) d++;
    printf("sz=%5llu off=+3 status=%d err=%s diff=%d\n",
           (unsigned long long)sz, (int)cbt->status(),
           cbt->error() ? cbt->error()->localizedDescription()->utf8String()
                        : "none",
           d);
    tb->release();
  }

  close(fd);
  free(ref);
  buf->release();
  fh->release();
  q->release();
  dev->release();
  return 0;
}
