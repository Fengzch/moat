# gfx936 HIP texture dimension guard

Environment: Hygon BW1000 (`gfx936`), Hygon DTK 26.04 at `/opt/dtk`.

Probe evidence: `hipGetDeviceProperties` reports `maxTexture3D=(16384,16384,8192)`. For a single-channel float 3D array and mipmapped array, width 16384 succeeds and width 16385 returns `invalid argument`.

Warp implementation: HIP `wp_texture_create_device` queries the active device and rejects any requested axis above its reported `maxTexture3D` before allocating. It preserves the requested shape and reports the offending dimension and device limit. CUDA code is unchanged.

Validation:

- DTK HIP build: `PATH=/root/.local/bin:$PATH bash utils/timeit.sh warp-1.19 compile -- python3 projects/warp-1.19/src/build_lib.py --hip --rocm-path /opt/dtk --hip-arch gfx936 --no-use-libmathdx --no-standalone --no-verbose -j 2` — PASS.
- Normal texture path: `PYTHONPATH=. python3 -m warp.tests -s autodetect -p test_texture.py -k test_texture1d_1channel_cuda_0 -j 1` — PASS on `cuda:0` / BW1000.
- Oversized texture: constructing `wp.Texture3D(np.zeros((4, 4, 16385), dtype=np.float32), device="cuda:0")` raises `RuntimeError: Failed to create CUDA texture: Warp error: HIP texture dimension 0 (16385) exceeds device limit (16384)` — PASS.
- HIP build option tests: `python3 -m unittest warp.tests.test_hip_build -v` — 7 PASS.
- MOAT fast gates: `PATH=/root/.local/bin:$PATH python3 utils/check.py --fast` — PASS; `published` is vacuous in this checkout.

This evidence is specific to Hygon DTK 26.04 on gfx936. It does not establish AMD ROCm behavior or resolve the separate explicit-LOD/mipmap and gradient failures. No commit, push, wheel, or release was created.
