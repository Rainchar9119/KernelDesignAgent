// Standalone benchmark for CUB/CCCL DeviceTopK::MaxKeys (float32), single-array primitive.
// CUB is single-array; batch is emulated by a host loop over rows on the same stream.
#include <cub/device/device_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/std/execution>
#include <cuda/stream>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x)                                                                             \
  do {                                                                                       \
    cudaError_t e = (x);                                                                     \
    if (e != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      std::exit(1);                                                                          \
    }                                                                                        \
  } while (0)

// Run MaxKeys once for one row (assumes temp_storage already sized/allocated).
static void run_maxkeys(void* d_temp, size_t temp_bytes, const float* d_in, float* d_out,
                        int num_items, int k, cudaStream_t stream) {
  auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                               cuda::execution::output_ordering::unsorted);
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, requirements};
  CHECK(cub::DeviceTopK::MaxKeys(d_temp, temp_bytes, d_in, d_out, num_items, k, env));
}

// Correctness: set of returned keys must equal set of true top-k (host partial_sort).
static bool verify(const std::vector<float>& host_in, const std::vector<float>& gpu_out, int num_items, int k) {
  std::vector<float> ref = host_in;
  std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), std::greater<float>());
  ref.resize(k);
  std::vector<float> got = gpu_out;
  std::sort(ref.begin(), ref.end());
  std::sort(got.begin(), got.end());
  // Compare as multisets. Values are unique random floats so exact match expected.
  for (int i = 0; i < k; ++i) {
    if (ref[i] != got[i]) return false;
  }
  return true;
}

int main() {
  const int Ls[]     = {2048, 8192, 32768, 131072, 262144};
  const int Ks[]     = {256, 512, 2048};
  const int batches[] = {1, 64, 256};

  const int warmup = 10;
  const int iters  = 50;

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev_start, ev_stop;
  CHECK(cudaEventCreate(&ev_start));
  CHECK(cudaEventCreate(&ev_stop));

  std::mt19937 rng(1234);

  std::printf("%-8s %-6s %-7s %-14s %-8s\n", "L", "K", "batch", "median_ms", "correct");
  std::printf("--------------------------------------------------------\n");

  for (int bi = 0; bi < 3; ++bi) {
    int B = batches[bi];
    for (int li = 0; li < 5; ++li) {
      int L = Ls[li];
      for (int ki = 0; ki < 3; ++ki) {
        int K = Ks[ki];
        if (K > L) continue;  // k capped to num_items; skip degenerate

        // Host input: B rows of L unique-ish floats.
        std::vector<float> h_in((size_t)B * L);
        std::uniform_real_distribution<float> dist(-1e4f, 1e4f);
        for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = dist(rng);

        float *d_in = nullptr, *d_out = nullptr;
        CHECK(cudaMalloc(&d_in, sizeof(float) * (size_t)B * L));
        CHECK(cudaMalloc(&d_out, sizeof(float) * (size_t)B * K));
        CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(float) * (size_t)B * L, cudaMemcpyHostToDevice));

        // Query temp storage (same size for every row since L,K fixed).
        size_t temp_bytes = 0;
        {
          auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                       cuda::execution::output_ordering::unsorted);
          auto env = cuda::std::execution::env{cuda::stream_ref{stream}, requirements};
          CHECK(cub::DeviceTopK::MaxKeys(nullptr, temp_bytes, d_in, d_out, L, K, env));
        }
        void* d_temp = nullptr;
        CHECK(cudaMalloc(&d_temp, temp_bytes));

        auto launch_all = [&]() {
          for (int b = 0; b < B; ++b) {
            run_maxkeys(d_temp, temp_bytes, d_in + (size_t)b * L, d_out + (size_t)b * K, L, K, stream);
          }
        };

        // Warmup
        for (int w = 0; w < warmup; ++w) launch_all();
        CHECK(cudaStreamSynchronize(stream));

        // Timed iters
        std::vector<float> times(iters);
        for (int it = 0; it < iters; ++it) {
          CHECK(cudaEventRecord(ev_start, stream));
          launch_all();
          CHECK(cudaEventRecord(ev_stop, stream));
          CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.f;
          CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          times[it] = ms;
        }
        std::sort(times.begin(), times.end());
        float median = times[iters / 2];

        // Correctness on row 0 (re-run once cleanly).
        launch_all();
        CHECK(cudaStreamSynchronize(stream));
        std::vector<float> h_out0(K);
        CHECK(cudaMemcpy(h_out0.data(), d_out, sizeof(float) * K, cudaMemcpyDeviceToHost));
        std::vector<float> row0(h_in.begin(), h_in.begin() + L);
        bool ok = verify(row0, h_out0, L, K);

        std::printf("%-8d %-6d %-7d %-14.5f %-8s\n", L, K, B, median, ok ? "PASS" : "FAIL");

        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
        CHECK(cudaFree(d_temp));
      }
    }
  }

  CHECK(cudaEventDestroy(ev_start));
  CHECK(cudaEventDestroy(ev_stop));
  CHECK(cudaStreamDestroy(stream));
  return 0;
}
