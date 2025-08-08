#include <fenv.h>

#include "stochrnddw.hpp"

/* Compute stochastic rounding of a+b. */
template <typename T> T _sr_add(const T a, const T b) {
  int prevround = fegetround();
  fesetround(FE_TONEAREST);

  // Compute floating-point approximation of sum and error.
  T s, t;
  two_sum(&s, &t, a, b);

  // Compute exponent with truncation.
  int exponent;
  fesetround(FE_TOWARDZERO);
  exponent =
      get_exponent(a + b); // Exponent of fraction / 2 (i.e. in [0.5, 1)).

  // Compute and renormalize a random number.
  T p = ldexp(SIGN(t) * (rand() / (T)RAND_MAX), exponent - 52);

  // Compute and return result.
  if (t >= 0)
    fesetround(FE_DOWNWARD);
  else
    fesetround(FE_UPWARD);

  T r = (t + p) + s;

  fesetround(prevround);
  return r;
}

/* Compute stochastic rounding of a*b. */
template <typename T> T _sr_mul_fma(const T a, const T b) {
  int prevround = fegetround();
  fesetround(FE_TOWARDZERO);

  // Compute floating-point approximation of product and error.
  T s, t;
  two_prod_fma(&s, &t, a, b);

  // Compute exponent with truncation.
  int exponent = get_exponent(s);

  // Compute and renormalize random number.
  T p = ldexp(SIGN(t) * (rand() / (T)RAND_MAX), exponent - 52);

  // Compute and return result.
  T r = (t + p) + s;

  fesetround(prevround);
  return r;
}

/* Compute stochastic rounding of a/b. */
template <typename T> T _sr_div(const T a, const T b) {
  int prevround = fegetround();
  fesetround(FE_TOWARDZERO);

  // Compute floating-point quotient, remainder, and residual.
  T s, t;
  s = a / b;
  t = fma(-s, b, a);
  t = t / b;

  // Compute exponent with truncation.
  int exponent = get_exponent(s);

  // Compute and renormalize random number.
  T p = ldexp(SIGN(t) * (rand() / (T)RAND_MAX), exponent - 52);

  // Compute and return result.
  T r = (t + p) + s;

  fesetround(prevround);
  return r;
}

/* Compute stochastic rounding of sqrt(a). */
template <typename T> T _sr_sqrt(const T a) {
  int prevround = fegetround();
  fesetround(FE_TOWARDZERO);

  // Compute floating-point quotient, remainder, and residual.
  T s, t;
  s = sqrt(a);
  t = fma(-s, s, a);
  t = t / (2 * s);

  // Compute exponent with truncation.
  int exponent = get_exponent(s);

  // Compute and renormalize random number.
  T p = ldexp(SIGN(t) * (rand() / (T)RAND_MAX), exponent - 52);

  // Compute and return result.
  T r = (t + p) + s;

  fesetround(prevround);
  return r;
}

template <> float sr_add(const float a, const float b) { return _sr_add(a, b); }
template <> double sr_add(const double a, const double b) { return _sr_add(a, b); }

template <> float sr_mul_fma(const float a, const float b) { return _sr_mul_fma(a, b); }
template <> double sr_mul_fma(const double a, const double b) { return _sr_mul_fma(a, b); }

template <> float sr_div(const float a, const float b) { return _sr_div(a, b); }
template <> double sr_div(const double a, const double b) { return _sr_div(a, b); }

template <> float sr_sqrt(const float a) { return _sr_sqrt(a); }
template <> double sr_sqrt(const double a) { return _sr_sqrt(a); }

