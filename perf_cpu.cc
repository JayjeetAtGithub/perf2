#include <chrono>
#include <iostream>

double measure_mips() {
  int64_t iterations = 1024 * 1024 * 1024;

  int32_t a = 46776;
  int32_t b = 34445;
  int32_t c = 63344;
  int32_t d = 75685;
  int32_t e = 19494;
  int32_t x = 0;

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < iterations; i++) {
    x = a + b + c + d + e + a + b + c + d + e + a + b + c + d + e + a + b + c +
        d + e + a + b + c + d + e + a + b + c + d + e + a + b + c + d + e + a +
        b + c + d + e + a + b + c + d + e + a + b + c + d + e;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  auto ips = ((double)(iterations * 49)) / ((double)(diff / 1e9));
  return (ips / 1e6);
}

double measure_flops() {
  int64_t iterations = 1024 * 1024 * 1024;

  float a = 46776.56857784;
  float b = 34445.14848484;
  float c = 63344.76857294;
  float d = 75685.83947567;
  float e = 19494.34848399;
  float x = 0.0;

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < iterations; i++) {
    x = a + b + c + d + e + a + b + c + d + e + a + b + c + d + e + a + b + c +
        d + e + a + b + c + d + e + a + b + c + d + e + a + b + c + d + e + a +
        b + c + d + e + a + b + c + d + e + a + b + c + d + e;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  auto flops = ((double)(iterations * 49)) / ((double)(diff / 1e9));
  return (flops / 1e9);
}

int main() {
  double gflops = measure_flops();
  double mips = measure_mips();

  std::cout << "GFLOPS: " << gflops << std::endl;
  std::cout << "MIPS: " << mips << std::endl;
}
