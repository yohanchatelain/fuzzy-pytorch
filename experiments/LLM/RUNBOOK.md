# Running the sweeps on a cluster node

The harness no longer carries `omp_ext`. PRISM reconciles a precision change
with every running thread through a configuration epoch, so `fuzzy_torch` sets
precision and rounding mode through PRISM's C API and the runs are no longer
pinned to one thread. A node therefore needs a PRISM new enough to have the
epoch, which the image it already has almost certainly does not.

## 1. One-off: refresh PRISM in the base image

PyTorch *calls* PRISM rather than inlining it -- every prism symbol in
`libtorch_cpu.so` is undefined and resolved at load time -- so replacing the
libraries is enough. PyTorch does not have to be rebuilt. This takes a few
minutes, against several hours for a full image build.

```bash
# on the node, with a PRISM checkout carrying the configuration epoch
git clone https://github.com/verificarlo/prism && cd prism
git checkout prism-config-epoch
"${CONTAINER_RUNTIME:-podman}" build \
    -f /path/to/pablo-fuzzy-pytorch/containers/prism-refresh/Containerfile \
    -t localhost/big-data-lab-team/fuzzy-pytorch:sr-epoch .
```

On slashbin set `CONTAINER_RUNTIME=docker`; it has no podman. The same variable
is read by `sweep_common.sh` and `run_remote_tmux.sh`.

The epoch has to be in that checkout. Until the branch is pushed, copy the
sources over instead of cloning:

```bash
tar -C ~/Work -cf - prism --exclude=.git --exclude=bazel-\* \
    | ssh slashbin 'tar -xf - -C ~/'
```

The build prints `PRISM refreshed: configuration epoch present` on success; it
fails rather than producing a quiet downgrade if the getters are missing.

If the node's LLVM differs from clang-20, pass `--build-arg CLANG=...` and
`--build-arg LLVM_BINDIR=...`. PRISM's `.bazelrc` and `config.bzl` pin an
absolute LLVM path, which the build rewrites.

## 2. Check the toolchain before spending node hours

```bash
podman run --rm -v "$PWD/python:/pkg:ro" -e PYTHONPATH=/pkg \
    -e VFC_BACKENDS_LOGGER=False -e VFC_BACKENDS=libinterflop_prism.so \
    localhost/big-data-lab-team/fuzzy-pytorch:sr-epoch \
    python3 -c "import fuzzy_torch; fuzzy_torch.assert_effective(); print('ok')"
```

`assert_effective` runs an operation through ATen at two precisions and fails
if the result does not move. A read-back through the C API is not enough: it
only proves Python and PRISM agree, not that PyTorch's arithmetic is the
arithmetic being configured. Note that a *one-element* tensor is not
instrumented -- it stays on a scalar path Verificarlo does not see -- so probes
must be wide.

## 3. Launch

`run_remote_tmux.sh` syncs `python/` and `experiments/LLM/`, builds the
experiment image from the repository root, and starts the sweep in a detached
tmux session:

```bash
./run_remote_tmux.sh <node> do_fine_perplexity.sh 256 sr all
./run_remote_tmux.sh <node> do_percomponent_perplexity.sh 256 sr
./run_remote_tmux.sh <node> do_fine_cumulative_perplexity.sh 256 sr
```

Environment knobs, all read by `sweep_common.sh`:

| Variable | Meaning |
|---|---|
| `MAX_JOBS` | concurrent containers |
| `THREADS_PER_RUN` | threads per run; empty means the container decides. Set to 1 to reproduce the single-threaded control |
| `T_MIN`, `T_MAX` | precision grid, default 4 to 14 |
| `SR_SEEDS`, `RN_SEEDS` | replication; SR defaults to 5 seeds, RN to 1 because PRISM's RN is bit-reproducible |
| `RESULTS_ROOT` | where logs go; point at shared storage |

## 4. Cost

One configuration is roughly nine minutes for a single 128-token window on a
laptop core. The unit-spaced grid multiplies the earlier configuration count by
about two: the per-component level is `3 groups x 11 precisions x 5 seeds` plus
a single-seed RN arm, so about 200 runs per context length, and the blockwise
and cumulative levels several times that. Budget accordingly and prefer
`MAX_JOBS` near the core count, since each run is now multi-threaded only if
`THREADS_PER_RUN` says so.

## 5. Controls worth running once

- **Threading**: the same configuration at `THREADS_PER_RUN=1` and unset. RN is
  bit-reproducible, so the two RN runs must agree exactly; SR must agree within
  the seed spread. This is what licenses dropping the single-thread pin.
- **Reference**: the `t=24` configuration under `VFC_BACKENDS=libinterflop_ieee.so`
  and under PRISM, which should agree to the reported precision.
