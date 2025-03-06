#pragma once

#include <chrono>
#include <immintrin.h>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#define ITERATIONS 10

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

static bool is_amxbf16_supported() {
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid"
                       : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                       : "a"(7), "c"(0));
  return edx & (1 << 22);
}

static void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  int32_t size = mem.get_desc().get_size();
  if (!handle)
    throw std::runtime_error("handle is nullptr.");
  uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
  if (!src)
    throw std::runtime_error("get_data_handle returned nullptr.");
  for (int32_t i = 0; i < size; ++i) {
    ((uint8_t *)handle)[i] = src[i];
  }
}

static void write_to_dnnl_memory(void const *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  int32_t size = mem.get_desc().get_size();
  if (!handle)
    throw std::runtime_error("handle is nullptr.");
  uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
  if (!dst)
    throw std::runtime_error("get_data_handle returned nullptr.");
  for (int32_t i = 0; i < size; ++i) {
    dst[i] = ((uint8_t *)handle)[i];
  }
}

static int64_t amx_matmul(int32_t const &r1, int32_t const &r2, const int32_t &c,
                       const __bf16 *a, const __bf16 *b, dnnl::engine &engine,
                       dnnl::stream &stream, bool debug) {
  dnnl::memory::dims a_dims = {r1, c};
  dnnl::memory::dims b_dims = {c, r2};
  dnnl::memory::dims c_dims = {r1, r2};

  auto a_md = dnnl::memory::desc(a_dims, dt::bf16, tag::ab);
  auto b_md = dnnl::memory::desc(b_dims, dt::bf16, tag::ab);
  auto c_md = dnnl::memory::desc(c_dims, dt::bf16, tag::ab);
  auto a_mem = dnnl::memory(a_md, engine);
  auto b_mem = dnnl::memory(b_md, engine);
  write_to_dnnl_memory(a, a_mem);
  write_to_dnnl_memory(b, b_mem);

  auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
  auto c_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::matmul(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, a_mem});
  args.insert({DNNL_ARG_WEIGHTS, b_mem});
  args.insert({DNNL_ARG_DST, c_mem});

  int64_t diff = 0;
  for (int32_t i = 0; i < ITERATIONS; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    prim.execute(stream, args);
    stream.wait();
    auto end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (debug) {
      std::cout << "matmul: dims: " << r1 << "," << r2 << "," << c << ": itr #" << i << " :"  << diff << " ns" << std::endl;
    }
  }
  return diff;
}

static int64_t amx_inner_product(int32_t const &n, int32_t const &oc,
                              int32_t const &ic, const __bf16 *s, const __bf16 *w,
                              dnnl::engine &engine, dnnl::stream &stream, bool debug) {

  dnnl::memory::dims s_dims = {n, ic};
  dnnl::memory::dims w_dims = {oc, ic};
  dnnl::memory::dims dst_dims = {n, oc};

  auto s_md = dnnl::memory::desc(s_dims, dt::bf16, tag::ab);
  auto w_md = dnnl::memory::desc(w_dims, dt::bf16, tag::ab);
  auto dst_md = dnnl::memory::desc(dst_dims, dt::bf16, tag::ab);
  
  auto s_mem = dnnl::memory(s_md, engine);
  auto w_mem = dnnl::memory(w_md, engine);
  write_to_dnnl_memory(s, s_mem);
  write_to_dnnl_memory(w, w_mem);

  auto pd = dnnl::inner_product_forward::primitive_desc(
      engine, dnnl::prop_kind::forward_training, s_md, w_md, dst_md);
  auto dst_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::inner_product_forward(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, s_mem});
  args.insert({DNNL_ARG_WEIGHTS, w_mem});
  args.insert({DNNL_ARG_DST, dst_mem});

  int64_t diff = 0;
  for (int32_t i = 0; i < ITERATIONS; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    prim.execute(stream, args);
    stream.wait();
    auto end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (debug) {
      std::cout << "ip: dims: " << n << "," << oc << "," << ic << ": itr #" << i << " :"  << diff << " ns" << std::endl;
    }
  }
  return diff;
}
