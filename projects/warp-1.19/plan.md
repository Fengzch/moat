# Warp 1.19 ROCm Release Plan

## Project

- Upstream: `https://github.com/NVIDIA/warp`, release branch `release-1.19`, baseline commit `dca5011a3fd9857756ccd9824c637ca6f8b436d6` (`v1.19.0`).
- Delivery: personal fork `https://github.com/Fengzch/warp`; pristine mirror `release-1.19`; port/release branch `release/1.19.0.amd0`.
- MOAT project: `warp-1.19`. This is the first Warp line recorded in this MOAT checkout, so it has no prior MOAT release dependency.

## Existing AMD Support

An authoritative AMD port exists at `https://github.com/AMD-Ecosystem/warp`, an official fork of NVIDIA/warp. Its `amd-integration` branch provides HIP compilation, HIPRTC, runtime dispatch, AMD-specific tests and ROCm build instructions. The organization also has a tagged `1.13.0+rocm.0` release line. Open PR [AMD-Ecosystem/warp#21](https://github.com/AMD-Ecosystem/warp/pull/21) is a large, still-open update of the gfx1151 HIP port to Warp 1.18.0; it is based on `amd-integration-halo` and has 556 changed files. NVIDIA-side WIP HIP shim PRs #1770 and #1932 are closed unmerged. Upstream 1.19 has no open AMD/HIP/ROCm PR found at planning time.

This is not a clean-room port. Start from the NVIDIA 1.19 pristine tree required by the release-line contract, then carry forward the AMD-owned HIP implementation and the current 1.18 update in reviewable increments. Do not merge the AMD integration branch wholesale without first reconciling its base: GitHub reports it diverged from NVIDIA `release-1.19` (57 commits ahead, 781 behind; merge base `d8f5e52d02990e21e2df4b67298423d7b98598fb`). Coordinate with AMD Warp maintainers on the intended 1.19 update source and preserve 1.19-only upstream changes.

The upstream 1.19 build is a Python/setuptools package with a custom native build driver (`build_lib.py`) and CMake for local native builds; it is not a PyTorch extension. The AMD implementation is also custom (`build_lib.py` detects ROCm and invokes HIP compilation), so use that working integration as the strategy template rather than a mechanical whole-tree hipify. Preserve the CUDA build path and retain AMD-specific code behind compiler/platform guards.

## Build Classification and Strategy

- Classification: custom Python/setuptools + CMake native runtime with runtime JIT/code generation; neither standard PyTorch-extension Strategy B nor a simple CMake-only Strategy A.
- Evidence: `pyproject.toml` uses `setuptools.build_meta`; `setup.py` packages the native binaries; `build_lib.py` locates CUDA and libmathdx and drives native compilation; root `CMakeLists.txt` enumerates the CUDA runtime sources and links CUDA Runtime/NVRTC/NVJitLink. Runtime kernel compilation is implemented in `warp/_src` and `warp/native`.
- Strategy: adapt the AMD fork's HIP build/runtime layer into the 1.19 source while keeping CUDA intact. Port each native CUDA TU with HIP compilation, provide HIP runtime/driver/NVRTC compatibility, and stage the resulting runtime library in the Python wheel using the existing package build hooks. Do not advertise `--no-cuda` as the HIP switch: in the current AMD build driver that flag disables HIP too. The ROCm build must detect ROCm while leaving the supported build mode enabled, or add an explicit HIP option if the 1.19 code requires it.
- MathDx: upstream build defaults to downloading NVIDIA libmathdx, while the AMD port supplies rocWMMA paths for selected tile operations and has unsupported/deferred tile cases. No NVIDIA libmathdx binary or proprietary implementation may be included in an AMD release without a recorded human legal ruling. First make ROCm builds independent of MathDx; then enumerate tile features that need an AMD implementation or an explicit scoped-out reason.

## CUDA Surface Inventory

- Native GPU translation units: 18 `.cu` files under `warp/native` (including `bvh`, `bvh_cubql`, `deterministic`, `hashgrid`, `mesh`, `reduce`, `runlength_encode`, `scan`, `sort`, `sparse`, `volume`, `volume_builder`, and `warp`). Preserve the Python-generated and C++ native runtime components that dispatch and load them.
- Runtime/driver surface: CUDA allocation and copies, device discovery/attributes, streams/events, graph capture, memory pools, texture/surface operations, module loading, and error handling are centralized in `warp/native/cuda_util.*` and consumed throughout the native layer. Map these to HIP runtime/driver APIs with explicit ownership/lifetime semantics.
- Runtime compilation: Warp generates kernels at runtime; CUDA uses NVRTC/PTX and associated compiler/linker libraries. Implement and test HIPRTC source compilation, include resolution, compile diagnostics, code-object loading, cache identity, and architecture selection. A successful AOT build alone is insufficient.
- Kernel primitives: inventory finds global/device code and launch syntax throughout native kernels; audit all uses of warp/tile collectives and literal 32 assumptions. Reconcile AMD's existing lane-width abstractions with the 1.19 tile implementation. Verify wave32 and wave64 independently; do not assume compilation proves subgroup correctness.
- Library dependencies: cuBLASDx/cuFFTDx/cuSOLVERDx are optional MathDx-backed tile features; upstream uses cuBQL and other vendored code in native spatial structures. Reuse the AMD cuBQL HIP adaptation and rocWMMA work where their APIs still match 1.19. Map remaining primitives to roc* libraries or implement a portable fallback; no CUDA-only library may remain on the ROCm path.
- Distribution surface: `build_lib.py`, `build_llvm.py`, `setup.py`, `pyproject.toml`, `warp/_src/build_dll.py`, generated headers, wheel metadata, and license manifests determine whether the installed package actually carries and loads the HIP runtime. Test a built wheel from a clean environment, not only an in-tree import.

## Risks

- HIPRTC differs from NVRTC in supported headers, options, diagnostics, and architecture naming. Runtime-compiled source/header staging and cache invalidation need direct tests.
- Wave width varies (wave64 on CDNA, wave32 on RDNA). Tile layout, shuffle, mask, reduction, and shared-memory sizing must be valid for both.
- AMD's current official port documents limited GPU/ROCm coverage and has active 1.18 update work. The selected linux-gfx936 host is a wave32 RDNA4 target not proven by the 1.13 release line; it must not be treated as a substitute for gfx942 wave64 or Windows validation.
- CUDA graphs, stream capture, allocator/mempool interop, texture behavior, and multiple-device tests are likely platform-sensitive. Preserve the fixes from the AMD implementation and validate their 1.19 equivalents.
- Tile MathDx operations do not have a blanket one-for-one ROCm library replacement. Validate every operation; keep correct portable fallback paths where possible and record deferred scope explicitly.
- `scan-nvidia` found proprietary-license text in upstream `README.md`, `licenses/cuda-LICENSE.txt`, and `licenses/libmathdx-LICENSE.txt`. The README explicitly says libmathdx is automatically fetched for source builds and statically included in prebuilt packages, under NVIDIA's Software License Agreement. A human/legal ruling is required on redistribution, removal of NVIDIA license text, and whether the release may omit MathDx. Do not publish a wheel or GitHub release while this is unresolved.
- NVIDIA's GitHub license metadata is Apache-2.0 and `LICENSE.md` is Apache-2.0, but that does not settle the separate license files. Top-level Apache-2.0 is not blanket clearance for the entire release artifact.

## File-by-File Change List

- `build_lib.py`, `warp/_src/build_dll.py`: ROCm discovery, architecture flags, HIP compiler invocation, conditional CUDA/HIP build selection, and safe packaging of the appropriate native runtime.
- `CMakeLists.txt`, `warp/native/cuda_util.h`, `warp/native/cuda_util.cpp`, and affected `warp/native/*.{cu,h,cpp}`: guarded HIP compatibility/runtime mappings and HIP compilation of the full native kernel surface; use AMD fork changes as reference and retain the CUDA path.
- Runtime code generation under `warp/_src/` and the HIP runtime/compiler compatibility headers: HIPRTC compile/link/load flow, diagnostics, include path, cache and code-object naming.
- Tile implementation headers and related Python dispatch: rocWMMA or portable fallbacks for supported tile kernels; explicitly scope MathDx-only operations that lack a supported equivalent until implemented.
- `setup.py`, `pyproject.toml`, generated package manifests, `README.md`, and `docs/user_guide/installation.rst`: build/test the ROCm wheel path, document supported hardware/toolkit and install commands, and ensure the installed artifact contains no unauthorized proprietary components.
- `warp/tests/` and `warp/tests/unittest_suites.py`: add/extend tests for HIP runtime compilation, core native kernels, graphs/streams, tiles, memory, and interop. Keep all CPU-only tests and CUDA tests available; do not silently remove coverage.
- `surface.json`: generated floor plus explicit runtime JIT, wheel contents, ROCm docs, benchmarks/examples, and license-sensitive package components.

## Build Commands

Use the AMD fork's build interface as the starting recipe; confirm the exact options against the 1.19-adapted build driver before implementation. On the current Linux host, build for its detected wave32 architecture (`gfx936`):

```bash
cd projects/warp-1.19/src
ROCM_PATH=/opt/rocm uv run build_lib.py --hip-arch gfx936 --no-use-libmathdx
```

The release build must also produce and install a wheel from a clean environment after the native runtime build. For other hosts, substitute their actual `CMAKE_HIP_ARCHITECTURES`/HIP architecture (e.g. a supported CDNA gfx9 target) and use the matching ROCm SDK. Do not claim the recipe is final until the 1.19-adapted build script accepts it and the wheel is tested.

## Test Plan

- Establish the upstream CPU-only baseline and preserve all non-GPU tests.
- Focused HIP tests: `uv run --extra dev -m warp.tests -s autodetect -p '*graph*.py'`, plus focused `*tile*`, `*stream*`, `*mesh*`, `*sparse*`, `*texture*`, `*deterministic*`, and runtime/JIT modules after confirming the suite's platform filters.
- Full supported suite: `uv run --extra dev -m warp.tests`; run it after building the native runtime, on each required platform. Record exact counts, skips and failures.
- Test HIPRTC dynamically generated kernels and clean-cache rebuilds; test Python wheel installation outside the source checkout to verify the packaged library and JIT headers are used.
- Required hardware evidence: one wave32 AMD GPU (current host gfx936 can provide the initial run), one wave64 CDNA GPU, and a Windows ROCm/HIP platform if Warp's 1.19 toolchain supports it. Each validator runs on real hardware; `MOAT_PLATFORM` is not validation.
- CUDA no-regression: compile the CUDA path with the pinned toolkit and architecture where the host supports it; do not infer CUDA behavior from HIP builds.
- Document unimplemented MathDx-only behavior as deferred/owed, never as silently covered.

## Open Questions

1. Obtain a human/legal ruling on the CUDA and libmathdx license material in the resulting source/wheel and whether MathDx must be excluded, replaced or separately licensed.
2. Coordinate with AMD-Ecosystem Warp maintainers on whether to use the open 1.18 update PR's source as the HIP baseline for 1.19, and how to avoid conflicting with their active release work.
3. Confirm which public API/features are required in the 1.19 release and whether MathDx-only tile functions may be explicitly deferred.
4. Confirm the supported ROCm versions and exact Windows ROCm status for Warp 1.19 before assigning gate completion.
