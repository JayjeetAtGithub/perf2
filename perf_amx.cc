#include "CLI11.hpp"
#include "VariadicTable.hpp"
#include "dist.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using pprinter =
    VariadicTable<std::string, std::string, double, double, double, double>;

#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define MILLION pow(10, 6)
#define BILLION pow(10, 9)

class Benchmark {
public:
  bool only_amx = false;
  dnnl::engine engine;
  dnnl::stream stream;
  pprinter *pt;
  std::vector<std::string> headers = {
      "Mode",       "N1 / N2 / M",   "Data size (MiB)",
      "Total FLOP", "Duration (us)", "GFLOPS"};

  Benchmark(dnnl::engine engine, dnnl::stream stream)
      : engine(engine), stream(stream) {
    pt = new pprinter(headers);
  }

  void print_results() {
    pt->print(std::cout);
    pt = new pprinter(headers);
  }

  void run_ip(uint64_t N1, uint64_t N2, uint64_t M) {
    std::vector<float> mat_a(N1 * M);
    std::vector<float> mat_b(N2 * M);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib;

    OMP_PARALLEL_FOR
    for (uint64_t i = 0; i < N1; i++) {
      for (uint64_t j = 0; j < M; j++) {
        mat_a[i * M + j] = distrib(rng);
      }
    }

    OMP_PARALLEL_FOR
    for (uint64_t i = 0; i < N2; i++) {
      for (uint64_t j = 0; j < M; j++) {
        mat_b[i * M + j] = distrib(rng);
      }
    }

    double data_size =
        ((double)(N1 * M * sizeof(float)) + (double)(N2 * M * sizeof(float))) / MILLION;
    uint64_t total_flop = (N1 * N2) * (2 * M - 1);
    std::string dims =
        std::to_string(N1) + "/" + std::to_string(N2) + "/" + std::to_string(M);

    if (!only_amx) {
      auto start = std::chrono::high_resolution_clock::now();
      for (uint64_t i = 0; i < N1; i++) {
        ip_distance_avx512(mat_a.data() + i * M, mat_b.data(), N2, M, engine,
                           stream);
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      double gflops =
          ((double)(total_flop / BILLION)) / ((double)(dur / MILLION));
      pt->addRow("IP / AVX512", dims, data_size, total_flop, dur, gflops);
    }

    {
      auto start = std::chrono::high_resolution_clock::now();
      amx_inner_product(N1, N2, M, mat_a.data(), mat_b.data(), engine, stream);
      auto end = std::chrono::high_resolution_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      double gflops =
          ((double)(total_flop / BILLION)) / ((double)(dur / MILLION));
      pt->addRow("IP / AMX", dims, data_size, total_flop, dur, gflops);
    }
  }

  void run_gemm(uint64_t N1, uint64_t N2, uint64_t M) {
    std::vector<float> mat_a(N1 * M);
    std::vector<float> mat_b(M * N2);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib;

    OMP_PARALLEL_FOR
    for (uint64_t i = 0; i < N1; i++) {
      for (uint64_t j = 0; j < M; j++) {
        mat_a[i * M + j] = distrib(rng);
      }
    }

    OMP_PARALLEL_FOR
    for (uint64_t i = 0; i < M; i++) {
      for (uint64_t j = 0; j < N2; j++) {
        mat_b[i * N2 + j] = distrib(rng);
      }
    }

    double data_size =
        ((double)(N1 * M * sizeof(float)) + (double)(M * N2 * sizeof(float))) / MILLION;
    uint64_t total_flop = (N1 * N2) * (2 * M - 1);
    std::string dims =
        std::to_string(N1) + "/" + std::to_string(N2) + "/" + std::to_string(M);

    {
      auto start = std::chrono::high_resolution_clock::now();
      amx_matmul(N1, N2, M, mat_a.data(), mat_b.data(), engine, stream);
      auto end = std::chrono::high_resolution_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      double gflops =
          ((double)(total_flop / BILLION)) / ((double)(dur / MILLION));
      pt->addRow("GEMM / AMX", dims, data_size, total_flop, dur, gflops);
    }
  }
};

void run_bench_sq_matrix() {
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  Benchmark bench(engine, stream);

  // Just bench AMX
  bench.only_amx = true;
  std::vector<uint64_t> sizes = {64,   128,  256,  512,   1024,
                                 2048, 4096, 8192, 16384, 32768};
  for (auto size : sizes) {
    bench.run_ip(size, size, size);
  }
  bench.print_results();
  for (auto size : sizes) {
    bench.run_gemm(size, size, size);
  }
  bench.print_results();
}

void run_bench_rect_matrix() {
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  Benchmark bench(engine, stream);

  uint64_t const n2 = 1024 * 1024 * 2;
  uint64_t const m = 1024 * 2;

  // Just bench AMX
  bench.only_amx = true;
  std::vector<uint64_t> n1s = {64,   128,  256,   512,   1024, 2048,
                               4096, 8192, 16384, 32768, 65536};
  for (auto n1 : n1s) {
    bench.run_ip(n1, n2, m);
  }
  bench.print_results();
}

int main(int argc, char **argv) {
  run_bench_sq_matrix();
  run_bench_rect_matrix();
}
