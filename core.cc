#include <chrono>
#include <iostream>

void measure_flops() {

  int64_t iterations = 1024 * 1024 * 1024;

  float a = 46776.56857784;
  float b = 34445.14848484;
  float c = 63344.76857294;
  float d = 75685.76857294;
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
  flops = flops / 1e9;
  std::cout << "GFlops: " << flops << std::endl;
}

int main() { measure_flops(); }
