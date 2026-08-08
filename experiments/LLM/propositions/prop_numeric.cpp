// Tests for the propositions of Section 4 that concern the reduction itself:
// unbiasedness (E1), concentration (E2), stagnation (E3), coherence (E4).
//
// These call PRISM's scalar API directly so the summation order is under our
// control, which matters because prop:unbiased claims to hold for any fixed
// summation tree.
//
//   g++ -O2 -std=c++17 prop_numeric.cpp -o prop_numeric \
//       -lprism-static -lhwy -L/usr/local/lib -Wl,-rpath,/usr/local/lib

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace prism::sr::scalar::static_dispatch {
float addf32(float, float);
}
namespace prism::sr {
extern thread_local int32_t virtual_precision_f32;
extern thread_local int32_t rounding_mode;
}
using prism::sr::scalar::static_dispatch::addf32;

constexpr int32_t SR = 0, RN = 1;

static void set_mode(int32_t t, int32_t mode) {
    prism::sr::virtual_precision_f32 = t;
    prism::sr::rounding_mode = mode;
}

// Round a float to the virtual grid of precision t, so that inputs are exactly
// representable and the test isolates accumulation from input rounding.
static float trunc_t(float x, int32_t t) {
    if (t >= 24) return x;
    uint32_t b;
    std::memcpy(&b, &x, 4);
    const uint32_t mask = ~((1u << (24 - t)) - 1u);
    b &= mask;
    float r;
    std::memcpy(&r, &b, 4);
    return r;
}

// Exact reference: Kahan in double. Inputs are binary32 of moderate magnitude,
// so this is exact far below the SR noise floor.
static double exact_sum(const std::vector<float> &c) {
    double s = 0.0, comp = 0.0;
    for (float v : c) {
        double y = double(v) - comp;
        double tt = s + y;
        comp = (tt - s) - y;
        s = tt;
    }
    return s;
}

enum class Order { LeftToRight, Reversed, PairwiseTree };

static float reduce(const std::vector<float> &c, Order order) {
    if (order == Order::LeftToRight) {
        float s = c[0];
        for (size_t i = 1; i < c.size(); ++i) s = addf32(s, c[i]);
        return s;
    }
    if (order == Order::Reversed) {
        float s = c.back();
        for (size_t i = c.size() - 1; i-- > 0;) s = addf32(s, c[i]);
        return s;
    }
    std::vector<float> buf(c);
    size_t n = buf.size();
    while (n > 1) {
        size_t m = 0;
        for (size_t i = 0; i + 1 < n; i += 2) buf[m++] = addf32(buf[i], buf[i + 1]);
        if (n % 2) buf[m++] = buf[n - 1];
        n = m;
    }
    return buf[0];
}

struct Stats {
    double mean, sd, z;   // z = (mean - ref) / (sd / sqrt(R))
    double rms;
};

static Stats run_trials(const std::vector<float> &c, double ref, int32_t t,
                        int32_t mode, Order order, int R) {
    set_mode(t, mode);
    double sum = 0.0, sumsq = 0.0, sqerr = 0.0;
    for (int r = 0; r < R; ++r) {
        double v = double(reduce(c, order));
        sum += v;
        sumsq += v * v;
        sqerr += (v - ref) * (v - ref);
    }
    double mean = sum / R;
    double var = sumsq / R - mean * mean;
    double sd = var > 0 ? std::sqrt(var) : 0.0;
    double se = sd / std::sqrt(double(R));
    return {mean, sd, se > 0 ? (mean - ref) / se : 0.0, std::sqrt(sqerr / R)};
}

static std::vector<float> make_vec(int n, int32_t t, uint64_t seed, double lo = 1.0,
                                   double hi = 2.0) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<float> c(n);
    for (int i = 0; i < n; ++i) c[i] = trunc_t(float(d(g)), t);
    return c;
}

// ------------------------------------------------------------------ E1
static void e1_unbiased() {
    printf("\n=== E1  prop:unbiased -- E[s_n] = s ===\n");
    printf("%-14s %5s %6s %7s %10s %14s %10s %8s\n", "order", "n", "t", "n*u",
           "rule", "mean-exact", "sd", "z");
    const int R = 200000;
    struct Cfg { int n; int32_t t; };
    Cfg cfgs[] = {{100, 12}, {1000, 8}, {4000, 6}};
    const char *onames[] = {"left-to-right", "reversed", "pairwise-tree"};
    Order orders[] = {Order::LeftToRight, Order::Reversed, Order::PairwiseTree};

    for (auto cfg : cfgs) {
        auto c = make_vec(cfg.n, cfg.t, 12345);
        double ref = exact_sum(c);
        double nu = cfg.n * std::pow(2.0, -cfg.t);
        for (int oi = 0; oi < 3; ++oi) {
            for (int32_t mode : {SR, RN}) {
                Stats s = run_trials(c, ref, cfg.t, mode, orders[oi], R);
                printf("%-14s %5d %6d %7.2f %10s %14.6f %10.4f %8.2f\n",
                       onames[oi], cfg.n, cfg.t, nu, mode == SR ? "SR" : "RN",
                       s.mean - ref, s.sd, s.z);
            }
        }
    }
    printf("  (|z| < 3 is consistent with zero bias at R=%d; RN is the control)\n", R);
}

// ------------------------------------------------------------------ E2
static void e2_concentration() {
    printf("\n=== E2  prop:azuma -- concentration, precondition n*u <= 1/2 ===\n");
    const int R = 100000;
    const double delta = 0.05;
    printf("%-5s %-5s %7s %13s %13s %10s %9s\n", "n", "t", "n*u", "bound",
           "max|err| SR", "violations", "tightness");
    struct Cfg { int n; int32_t t; };
    for (auto cfg : {Cfg{100, 12}, Cfg{100, 10}, Cfg{50, 8}}) {
        auto c = make_vec(cfg.n, cfg.t, 999);
        double ref = exact_sum(c);
        double u = std::pow(2.0, -cfg.t);
        double nu = cfg.n * u;

        double sum_c2 = 0.0;
        for (float v : c) sum_c2 += double(v) * double(v);
        double A = 0.0, sum_A2 = 0.0;
        for (int j = 0; j < cfg.n; ++j) {
            A += std::fabs(double(c[j]));
            if (j >= 1) sum_A2 += A * A;
        }
        double bound = 2.0 * u * std::exp(2.0 * nu) *
                       std::sqrt(2.0 * std::log(2.0 / delta) * (sum_c2 + sum_A2));

        set_mode(cfg.t, SR);
        int viol = 0;
        double maxerr = 0.0;
        for (int r = 0; r < R; ++r) {
            double e = std::fabs(double(reduce(c, Order::LeftToRight)) - ref);
            if (e > bound) ++viol;
            if (e > maxerr) maxerr = e;
        }
        printf("%-5d %-5d %7.3f %13.5f %13.5f %9.4f%% %9.2f\n", cfg.n, cfg.t, nu,
               bound, maxerr, 100.0 * viol / R, bound / maxerr);
    }
    printf("  (violation rate should be <= %.0f%%; tightness = bound / max observed)\n",
           100 * delta);

    // Error growth with n. RN's error is deterministic for a given vector, so a
    // single vector per n samples one realization and is far too noisy to read a
    // scaling exponent from; averaging over many *vectors* is required. SR is
    // averaged over both vectors and draws.
    printf("\n  -- error growth with n, averaged over %d vectors, t=10 --\n", 200);
    printf("%8s %14s %14s %10s %10s\n", "n", "rms SR", "mean|err| RN", "SR/sqrt(n)",
           "RN/n");
    const int V = 200, Rsr = 300;
    for (int n : {32, 64, 128, 256, 512, 1024}) {
        double acc_sr = 0.0, acc_rn = 0.0;
        for (int v = 0; v < V; ++v) {
            auto c = make_vec(n, 10, 4242 + 7919ull * v);
            double ref = exact_sum(c);
            Stats sr = run_trials(c, ref, 10, SR, Order::LeftToRight, Rsr);
            Stats rn = run_trials(c, ref, 10, RN, Order::LeftToRight, 1);
            acc_sr += sr.rms;
            acc_rn += std::fabs(rn.mean - ref);
        }
        double rs = acc_sr / V, rr = acc_rn / V;
        printf("%8d %14.6f %14.6f %10.5f %10.5f\n", n, rs, rr,
               rs / std::sqrt(double(n)), rr / double(n));
    }
    printf("  (if SR ~ sqrt(n)u and RN ~ n u, the last two columns are constant)\n");
}

// ------------------------------------------------------------------ E3
static void e3_stagnation() {
    printf("\n=== E3  prop:stagnation -- swamping window S/mu in [1/u, 2/u] ===\n");
    printf("%-5s %10s %12s %12s %14s %12s\n", "t", "1/u = 2^t", "2/u", "RN stalls at",
           "in window?", "SR mean/exact");
    const float mu = 1.0f;
    for (int32_t t : {6, 8, 10, 12}) {
        set_mode(t, RN);
        float s = 0.0f;
        long k = 0;
        const long kmax = 1L << 20;
        while (k < kmax) {
            float ns = addf32(s, mu);
            if (ns == s) break;
            s = ns;
            ++k;
        }
        double lo = std::pow(2.0, t), hi = 2.0 * std::pow(2.0, t);
        bool inwin = (double(s) >= lo * 0.999) && (double(s) <= hi * 1.001);

        // Same accumulation under SR, averaged: the claim is it stays unbiased.
        const int R = 2000;
        const long n = long(hi) * 2;
        set_mode(t, SR);
        double acc = 0.0;
        for (int r = 0; r < R; ++r) {
            float ss = 0.0f;
            for (long i = 0; i < n; ++i) ss = addf32(ss, mu);
            acc += double(ss);
        }
        printf("%-5d %10.0f %12.0f %12.0f %14s %12.5f\n", t, lo, hi, double(s),
               inwin ? "yes" : "NO", (acc / R) / double(n));
    }
    printf("  (RN stall value should land in [1/u, 2/u]; SR ratio should be ~1)\n");
}

// ------------------------------------------------------------------ E4
static void e4_coherence() {
    printf("\n=== E4  eq:coherence -- kappa in [1, sqrt(n-1)] ===\n");
    // t = 8 rather than 10: at t = 10 a length-200 sum of unit terms is exactly
    // representable throughout, so no rounding occurs and both rules are exact.
    // The comparison is only meaningful where the grid actually bites.
    const int n = 200, t = 8, R = 40000;
    printf("%-22s %9s %9s %12s %12s %12s\n", "construction", "kappa",
           "sqrt(n-1)", "|RN bias|", "SR sd", "ratio");

    auto analyse = [&](const char *name, std::vector<float> c) {
        double ref = exact_sum(c);
        // kappa over partial sums s_2..s_n
        double l1 = 0.0, l2 = 0.0, ps = double(c[0]);
        for (int j = 1; j < n; ++j) {
            ps += double(c[j]);
            l1 += std::fabs(ps);
            l2 += ps * ps;
        }
        double kappa = l1 / std::sqrt(l2);
        Stats rn = run_trials(c, ref, t, RN, Order::LeftToRight, 1);
        Stats sr = run_trials(c, ref, t, SR, Order::LeftToRight, R);
        double bias = std::fabs(rn.mean - ref);
        printf("%-22s %9.3f %9.3f %12.6f %12.6f %12.3f\n", name, kappa,
               std::sqrt(double(n - 1)), bias, sr.sd,
               sr.sd > 0 ? bias / sr.sd : 0.0);
    };

    // (a) comparable partial sums: all terms positive, so the partial sums grow
    // monotonically and are all of the same order.
    analyse("comparable sums", make_vec(n, t, 7, 1.0, 2.0));
    // (b) one dominant partial sum: a large spike that is immediately cancelled
    {
        std::vector<float> c(n, trunc_t(1e-3f, t));
        c[n / 2] = trunc_t(1e3f, t);
        c[n / 2 + 1] = trunc_t(-1e3f, t);
        analyse("one dominant sum", c);
    }
    // (c) intermediate: alternating signs, partial sums oscillate near zero
    {
        std::vector<float> c = make_vec(n, t, 11);
        for (int i = 1; i < n; i += 2) c[i] = -c[i];
        analyse("alternating signs", c);
    }
    printf("  (kappa is a bound on exposure; the realized ratio need not equal it)\n");
}

int main(int argc, char **argv) {
    std::string which = argc > 1 ? argv[1] : "all";
    if (which == "all" || which == "e1") e1_unbiased();
    if (which == "all" || which == "e2") e2_concentration();
    if (which == "all" || which == "e3") e3_stagnation();
    if (which == "all" || which == "e4") e4_coherence();
    return 0;
}
