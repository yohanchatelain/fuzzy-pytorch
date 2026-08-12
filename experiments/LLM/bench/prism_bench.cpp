// Throughput of PRISM's array interface relative to native arithmetic, in both
// rounding modes.
//
// The existing harness in tests/vector/test_sr_performance.cpp benchmarks SR
// only. The SR-versus-RN comparison needs both, because that pair isolates the
// cost of the random draw: the two modes run the same kernel and differ only in
// whether the threshold is drawn or fixed at 1/2.
//
// Build against libprism-static.so or libprism-dynamic.so to get the
// corresponding dispatch mode.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The dispatch namespace is chosen by which library we link against, so it is a
// compile-time parameter here: -DDISPATCH=static_dispatch with libprism-static.
#ifndef DISPATCH
#define DISPATCH dynamic_dispatch
#endif

namespace prism::sr::vector::DISPATCH::variable {
void addf32(const float *, const float *, float *, size_t);
void mulf32(const float *, const float *, float *, size_t);
void divf32(const float *, const float *, float *, size_t);
void fmaf32(const float *, const float *, const float *, float *, size_t);
void addf64(const double *, const double *, double *, size_t);
void mulf64(const double *, const double *, double *, size_t);
void divf64(const double *, const double *, double *, size_t);
void fmaf64(const double *, const double *, const double *, double *, size_t);
} // namespace prism::sr::vector::DISPATCH::variable

extern "C" {
void interflop_prism_set_rounding_mode(int32_t mode);
}

namespace pd = prism::sr::vector::DISPATCH::variable;

constexpr int32_t SR = 0;
constexpr int32_t RN = 1;

static size_t N = 1 << 16;   // elements per call
static int REPS = 200;       // calls per timing

template <typename F> static double time_ns_per_elem(F &&f) {
    f();  // warm up: first touch pays TLS init and page faults
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < REPS; ++r) f();
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / (double(REPS) * double(N));
}

template <typename T> struct Buf {
    std::vector<T> a, b, c, out;
    Buf() : a(N), b(N), c(N), out(N) {
        for (size_t i = 0; i < N; ++i) {
            a[i] = T(1.0) + T(i % 97) / T(97.0);
            b[i] = T(1.0) + T(i % 89) / T(89.0);
            c[i] = T(1.0) + T(i % 83) / T(83.0);
        }
    }
};

int main(int argc, char **argv) {
    if (argc > 1) N = std::strtoul(argv[1], nullptr, 10);
    if (argc > 2) REPS = std::atoi(argv[2]);

    Buf<float> f;
    Buf<double> d;

    // Native baselines. Kept in the same shape as the PRISM calls so the
    // comparison is per-element throughput of an elementwise loop either way.
    double nat_add32 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) f.out[i] = f.a[i] + f.b[i];
    });
    double nat_mul32 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) f.out[i] = f.a[i] * f.b[i];
    });
    double nat_div32 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) f.out[i] = f.a[i] / f.b[i];
    });
    double nat_fma32 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) f.out[i] = f.a[i] * f.b[i] + f.c[i];
    });
    double nat_add64 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) d.out[i] = d.a[i] + d.b[i];
    });
    double nat_mul64 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) d.out[i] = d.a[i] * d.b[i];
    });
    double nat_div64 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) d.out[i] = d.a[i] / d.b[i];
    });
    double nat_fma64 = time_ns_per_elem([&] {
        for (size_t i = 0; i < N; ++i) d.out[i] = d.a[i] * d.b[i] + d.c[i];
    });

    printf("# N=%zu reps=%d\n", N, REPS);
    printf("# %-6s %-8s %12s %12s %12s %10s %10s\n", "op", "type",
           "native_ns", "sr_ns", "rn_ns", "sr/nat", "rn/nat");

    auto row = [&](const char *op, const char *ty, double nat,
                   auto &&call) {
        interflop_prism_set_rounding_mode(SR);
        double sr = time_ns_per_elem(call);
        interflop_prism_set_rounding_mode(RN);
        double rn = time_ns_per_elem(call);
        printf("  %-6s %-8s %12.4f %12.4f %12.4f %10.1f %10.1f\n",
               op, ty, nat, sr, rn, sr / nat, rn / nat);
    };

    row("add", "binary32", nat_add32,
        [&] { pd::addf32(f.a.data(), f.b.data(), f.out.data(), N); });
    row("mul", "binary32", nat_mul32,
        [&] { pd::mulf32(f.a.data(), f.b.data(), f.out.data(), N); });
    row("div", "binary32", nat_div32,
        [&] { pd::divf32(f.a.data(), f.b.data(), f.out.data(), N); });
    row("fma", "binary32", nat_fma32,
        [&] { pd::fmaf32(f.a.data(), f.b.data(), f.c.data(), f.out.data(), N); });
    row("add", "binary64", nat_add64,
        [&] { pd::addf64(d.a.data(), d.b.data(), d.out.data(), N); });
    row("mul", "binary64", nat_mul64,
        [&] { pd::mulf64(d.a.data(), d.b.data(), d.out.data(), N); });
    row("div", "binary64", nat_div64,
        [&] { pd::divf64(d.a.data(), d.b.data(), d.out.data(), N); });
    row("fma", "binary64", nat_fma64,
        [&] { pd::fmaf64(d.a.data(), d.b.data(), d.c.data(), d.out.data(), N); });
    return 0;
}
