# omp_ext

Lets the per-module precision hooks in `eval_utils.py` actually change PRISM's
virtual precision and rounding mode during a forward pass.

## Why it is needed

PRISM keeps virtual precision and rounding mode in thread-local storage
(`prism::sr::virtual_precision_f32`, `::virtual_precision_f64`,
`::rounding_mode`). Two things follow.

**The public C API cannot change precision mid-run.** The only exported
precision setter, `interflop_prism_set_default_virtual_precision_binary32`,
writes the *process-wide default*. A thread copies that default into its TLS on
first use, so once a thread has executed instrumented arithmetic the default no
longer reaches it — the setter becomes a silent no-op for that thread, while the
getter still reports the new value. The main Python thread has already run
arithmetic by the time the model finishes loading. Changing precision per module
therefore requires writing the TLS directly, which is what this module does.

**A write reaches only the calling thread.** ATen dispatches operators across an
OpenMP team, so each setter runs inside a `#pragma omp parallel` region to reach
every worker in the current team, and also updates the process-wide default so
threads created later inherit the same setting.

## Build

```bash
cd omp_ext && python3 setup.py build_ext --inplace
```

Then put the directory on `PYTHONPATH`. The `Containerfile` does both.

`setup.py` reads `libtorch_cpu.so`'s dynamic section to find which PRISM build it
links against (`libprism-static.so` or `libprism-dynamic.so`) and links the same
one. Binding to the other would give the extension its own TLS block, and every
precision change would be silently ignored.

## Requirements

A Verificarlo-instrumented PyTorch built against a PRISM that exports the
rounding-mode API. Older builds export `virtual_precision_*` but not
`rounding_mode`, and importing the module there fails with

```
ImportError: undefined symbol: _ZN5prism2sr21default_rounding_modeE
```

which means the image predates untied-RN support and has to be rebuilt.

## API

| Call | Effect |
|---|---|
| `set_precision(t)` | Virtual precision for binary32 and binary64, `2 <= t <= 24` |
| `set_rounding_mode(mode)` | `omp_ext.SR` (0) or `omp_ext.RN` (1) |
| `get_precision()` / `get_precision_binary64()` | Calling thread's virtual precision |
| `get_rounding_mode()` | Calling thread's rounding mode |
| `num_broadcast_threads()` | Workers a setter reaches; compare against `omp_get_max_threads()` |

Setters raise on out-of-range input, and raise if a write does not read back —
the symptom of being bound to the wrong PRISM library. `eval_utils.set_precision`
and `set_rounding_mode` re-check the round trip on every call, so a
silently-ignored precision change fails the run instead of producing a
plausible-looking perplexity at full precision.
