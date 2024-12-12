#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <immintrin.h>

#define CACHE_LINE_SIZE 16

void read_c(std::vector<std::vector<int32_t>> &v) {
    for (int32_t i = 0; i < v.size(); i++) {
        for (int32_t j = 0; j < CACHE_LINE_SIZE; j++) {
            v[i][j] = v[i][j] + 1;
        }
    }
}

void read_cu(std::vector<std::vector<int32_t>> &v) {
    for (int32_t j = 0; j < CACHE_LINE_SIZE; j++) {
        for (int32_t i = 0; i < v.size(); i++) {
             v[i][j] = v[i][j] + 1;
        }
    }
}

int main(int argc, char* argv[]) {
    int32_t n = atoi(argv[1]);

    std::vector<std::vector<int32_t>> v(n, std::vector<int32_t>(16, 0));
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < CACHE_LINE_SIZE; j++) {
            v[i][j] = i + j;
        }
    }

    int32_t bytes = n * sizeof(int32_t);
    std::cout << "size of v (KiB): " << ((double)(bytes) / (1024)) << " KiB" << std::endl;
    std::cout << "size of v (MiB): " << ((double)(bytes) / (1024 * 1024)) << " MiB" << std::endl;

    for (int32_t r = 0; r < 5; r++) {
        auto t1 = std::chrono::high_resolution_clock::now();    
        read_c(v);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto diff =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "time to read cache friendly: " << diff << " us" << std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        read_cu(v);
        t2 = std::chrono::high_resolution_clock::now();
        auto diff2 =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "time to read cache unfriendly: " << diff2 << " us" << std::endl;

        for (int32_t i = 0; i < v.size(); i++) {
            _mm_prefetch(&v[i][0], _MM_HINT_T0);
        }
        
        t1 = std::chrono::high_resolution_clock::now();
        read_c(v);
        t2 = std::chrono::high_resolution_clock::now();
        auto diff3 =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "time to read cache prefetch: " << diff3 << " us" << std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        read_cu(v);
        t2 = std::chrono::high_resolution_clock::now();
        auto diff4 =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "time to read cache unfriendly prefetch: " << diff4 << " us" << std::endl;
    }

    return 0;
}
