// Broadcasts PRISM's virtual precision and rounding mode into every OpenMP
// worker slot, so that a forward pre-hook can lower precision for the duration
// of one module and a forward hook can restore it afterwards.
//
// PRISM keeps both settings in thread-local storage
// (prism::sr::virtual_precision_f32, ::virtual_precision_f64,
// ::rounding_mode).  Two consequences shape this module:
//
//   * The public C API exposes setters for the *process-wide defaults* only.  A
//     thread copies the default into its TLS on first use, so changing a default
//     after a thread has executed instrumented arithmetic does not reach that
//     thread.  Changing precision mid-forward-pass therefore has to write the
//     TLS directly.  The defaults are updated as well, so that threads created
//     later inherit the same setting.
//   * A write reaches only the calling thread, while ATen dispatches operators
//     across an OpenMP team.  Each setter runs inside a parallel region so every
//     worker in the current team is updated.
//
// This module must bind to the same libprism-*.so that libtorch_cpu.so links
// against, or it writes a TLS block the instrumented arithmetic never reads.
// setup.py resolves that from libtorch_cpu.so's NEEDED entries.

#include <torch/extension.h>

#include <omp.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace prism {
namespace sr {
extern thread_local int32_t virtual_precision_f32;
extern thread_local int32_t virtual_precision_f64;
extern thread_local int32_t rounding_mode;
extern int32_t default_rounding_mode;
} // namespace sr
} // namespace prism

extern "C" {
void interflop_prism_set_default_virtual_precision_binary32(int32_t t);
void interflop_prism_set_default_virtual_precision_binary64(int32_t t);
int32_t interflop_prism_get_default_virtual_precision_binary32(void);
}

namespace {

// binary32 significand width; the sweeps never exceed it, and PRISM asserts on
// anything outside [2, precision].
constexpr int32_t kPrecisionMin = 2;
constexpr int32_t kPrecisionMax = 24;

constexpr int32_t kModeSR = 0;
constexpr int32_t kModeRN = 1;

// Applies to both binary32 and binary64: an instrumented forward pass may touch
// either, and the sweeps describe a single virtual precision for the whole
// forward pass.
void set_precision(int32_t t) {
  if (t < kPrecisionMin || t > kPrecisionMax) {
    throw std::invalid_argument("omp_ext.set_precision: precision " +
                                std::to_string(t) + " outside [" +
                                std::to_string(kPrecisionMin) + ", " +
                                std::to_string(kPrecisionMax) + "]");
  }

  interflop_prism_set_default_virtual_precision_binary32(t);
  interflop_prism_set_default_virtual_precision_binary64(t);

#pragma omp parallel
  {
    prism::sr::virtual_precision_f32 = t;
    prism::sr::virtual_precision_f64 = t;
  }

  // The calling thread is the one that runs the operators under
  // OMP_NUM_THREADS=1, so a mismatch here means the module is bound to a
  // different TLS block than the instrumented arithmetic reads.
  if (prism::sr::virtual_precision_f32 != t) {
    throw std::runtime_error(
        "omp_ext.set_precision: wrote " + std::to_string(t) +
        " but read back " + std::to_string(prism::sr::virtual_precision_f32) +
        "; omp_ext is not bound to the PRISM library used by libtorch_cpu.so");
  }
}

void set_rounding_mode(int32_t mode) {
  if (mode != kModeSR && mode != kModeRN) {
    throw std::invalid_argument(
        "omp_ext.set_rounding_mode: mode must be 0 (SR) or 1 (RN), got " +
        std::to_string(mode));
  }

  prism::sr::default_rounding_mode = mode;

#pragma omp parallel
  { prism::sr::rounding_mode = mode; }

  if (prism::sr::rounding_mode != mode) {
    throw std::runtime_error(
        "omp_ext.set_rounding_mode: wrote " + std::to_string(mode) +
        " but read back " + std::to_string(prism::sr::rounding_mode) +
        "; omp_ext is not bound to the PRISM library used by libtorch_cpu.so");
  }
}

int32_t get_precision() { return prism::sr::virtual_precision_f32; }

int32_t get_precision_binary64() { return prism::sr::virtual_precision_f64; }

int32_t get_rounding_mode() { return prism::sr::rounding_mode; }

// Number of OpenMP workers a setter reaches.  A run that expects thread-scoped
// precision to be meaningful should see this match omp_get_max_threads().
int32_t num_broadcast_threads() {
  int32_t n = 0;
#pragma omp parallel reduction(+ : n)
  { n += 1; }
  return n;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Broadcast PRISM virtual precision and rounding mode across OpenMP "
            "threads";
  m.def("set_precision", &set_precision, py::arg("precision"),
        "Set PRISM virtual precision (binary32 and binary64) on every OpenMP "
        "worker and on the process-wide default");
  m.def("set_rounding_mode", &set_rounding_mode, py::arg("mode"),
        "Set PRISM rounding mode (0 = SR, 1 = RN) on every OpenMP worker and "
        "on the process-wide default");
  m.def("get_precision", &get_precision,
        "Virtual binary32 precision of the calling thread");
  m.def("get_precision_binary64", &get_precision_binary64,
        "Virtual binary64 precision of the calling thread");
  m.def("get_rounding_mode", &get_rounding_mode,
        "Rounding mode of the calling thread (0 = SR, 1 = RN)");
  m.def("num_broadcast_threads", &num_broadcast_threads,
        "Number of OpenMP workers a setter reaches");
  m.attr("SR") = kModeSR;
  m.attr("RN") = kModeRN;
}
