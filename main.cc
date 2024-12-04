#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>

int main(int argc, char* argv[]) {
    int64_t n = atoi(argv[1]);
    int64_t *v = new int64_t[n];

    auto s = std::chrono::high_resolution_clock::now();    
    for (int64_t i = 0; i < n; i++) {
        v[i] = i;
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    int64_t bytes = n * sizeof(int64_t);
    std::cout << "size of v (KiB): " << ((double)(bytes) / (1024)) << " KiB" << std::endl;
    std::cout << "size of v (MiB): " << ((double)(bytes) / (1024 * 1024)) << " MiB" << std::endl;

    std::cout << "time to fill v: " << diff.count() * 1000000 << " us" << std::endl;

    for (int64_t r = 0; r < 50; r++) {
        s = std::chrono::high_resolution_clock::now();    
        for (int64_t i = 0; i < n; i++) {
            int64_t x = v[i];
        }
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        std::cout << "time to read v: " << diff.count() * pow(10, 6) << " us" << std::endl;
        std::cout << "time to read a single byte: " << (diff.count() * pow(10, 9)) / (bytes) << " ns" << std::endl;
    }

    delete[] v;
    return 0;
}
