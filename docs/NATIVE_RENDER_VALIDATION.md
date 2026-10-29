# Native RTX 6000 Ada render validation

Run stock Blender directly on the workstation. This runner needs Blender, an NVIDIA
driver and host Python with NumPy/Pillow; no extra rendering service or plugin is
required. It opens the full Blender GUI and starts a real F12 render. Each scene
and configuration gets a fresh process, sequentially. It saves Render Result PNGs;
it does not take screenshots or save source scenes or user preferences.

## Get the code

```powershell
git fetch origin
git switch --track origin/native-render-validation
python -m pip install -r requirements-native-render.txt
```

If the local branch already exists, use `git switch native-render-validation` and
`git pull --ff-only` instead. Use Blender **4.5.15** for the same build as the current
reference images. Put the portable build at `blender/4.5.15/blender.exe`, or pass
`--blender "D:\Apps\Blender 4.5\blender.exe"` instead of `--blender-version`.

If needed, the existing setup downloads that build and the Classroom smoke scene:

```powershell
python setup_viewport_test.py --builds 4.5.15 --scene smoke
```

Copy your own models into `models/`. Models and results are deliberately excluded
from Git. Keep companion textures, caches and linked files in their supplied
relative folders. In particular, copy the whole `stress_model` directory, including
`textures`, alongside `water_tents_desert.blend`. A `.blend1` backup is not needed.

## Water Tents Desert

```powershell
python run_native_render_suite.py `
  --blender-version 4.5.15 `
  --model .\models\stress_model\water_tents_desert.blend `
  --profile authored-look `
  --gpu "RTX 6000" --gpu-index 0 `
  --enable-autoexec `
  --output .\results\native_water_tents
```

This runs all three cases:

| Case | Compute | Denoising / raster settings |
|---|---|---|
| `cycles-cuda-oidn-gpu` | CUDA, GPU Compute | GPU OpenImageDenoise, High quality where available, Accurate, Albedo + Normal |
| `cycles-optix-optix` | OptiX, GPU Compute | OptiX denoising |
| `eevee` | OpenGL adapter | 64 samples, Screen-Trace ray tracing 1:1, shadows 4 rays / 12 steps, volume shadows on |

For this scene, the authored-look profile retains **Cam_Hero**, frame **1**, the
**Desert Sky (golden hour)** world, **AgX Medium High Contrast**, exposure **−3**,
and the fog/heat-haze compositor. Cycles bounce limits remain
`[12, 4, 4, 12, 2, 8]` (total, diffuse, glossy, transmission, volume, transparent).

The reduced settings match the current visual smoke comparisons: **480×270**
(stored as 1920×1080 at 25%), **16 fixed Cycles samples**, adaptive sampling off,
seed 0, denoising on, persistent data on, motion blur/simplify off. Camera DOF,
world, materials and compositor remain authored. The file's original 4K/1024-sample
adaptive setup is not used for this smoke run. This is not a performance protocol.

`--enable-autoexec` matches the reference-run launch policy. Use it only for scene
files you have reviewed/trust. Omit it to keep embedded scripts/drivers disabled;
a scene that requires them fails explicitly rather than silently changing behavior.

## Medieval fisher hut and Lone Monk

Use the `smoke` profile to match their earlier comparison settings:

```powershell
python run_native_render_suite.py `
  --blender-version 4.5.15 `
  --model .\models\medieval_fisher_hut.blend `
  --model .\models\archviz\lone_monk.blend `
  --profile smoke `
  --gpu "RTX 6000" --gpu-index 0 `
  --enable-autoexec `
  --output .\results\native_hut_lone_monk
```

This runs three cases per scene. The earlier comparisons contain stock Cycles
references for both API/denoiser pairs; EEVEE is an additional separate image.
The smoke profile sets **AgX / Look None / exposure 0 / gamma 1** and Cycles bounce
limits `[16, 4, 4, 12, 0, 16]`. Other shared smoke settings are as above. The authored
active camera and frame are retained. For exact comparisons, use the same original
scene files and check source SHA-256, camera, frame and settings in the result JSON.

## Two GPUs on the workstation

**Cycles defaults to one GPU**, not both. `--gpu-index 0` selects the first matching
device sorted by its ID within each API; `--gpu-index 1` selects the second. CPU and
all other API/device entries are disabled. Selected device names and IDs are
recorded before, during and after rendering. Check the IDs to identify the adapter.

Use `--all-matching-gpus` only for an intentional multi-GPU baseline, and label it
accordingly. Such a run is not equivalent to a single-GPU result.

**EEVEE uses the OpenGL adapter chosen by Windows/the driver.** The Cycles GPU index
does not select EEVEE's physical GPU. The worker verifies an RTX 6000 OpenGL renderer
and records its renderer/vendor strings. On a two-identical-GPU host, confirm the
physical adapter with your workstation GPU telemetry when that distinction matters.
The renderer string alone cannot distinguish the two identical cards.

## Options and outputs

```powershell
# Validate paths and see exact commands without launching Blender:
python run_native_render_suite.py --model .\models\stress_model\water_tents_desert.blend --dry-run

# Run only one case, with a larger process timeout:
python run_native_render_suite.py --model .\models\stress_model\water_tents_desert.blend `
  --cases cycles-optix-optix --timeout 1200 --output .\results\native_water_optix

# Other dimensions/sample counts are supported, but change comparison conditions:
python run_native_render_suite.py --model .\models\stress_model\water_tents_desert.blend `
  --width 1920 --height 1080 --samples 128 --output .\results\native_water_larger
```

Default process timeout: 600 seconds, including source loading and shader setup.
Use a new output directory per run; existing results are never overwritten or
silently reused. Unsupported GPUs/settings, process crashes and timeouts are
reported as failures; no cases are skipped. Exit code 0 means every requested case
passed; 1 means a case failed/interrupted; 2 means invalid arguments.

Open `report.html` in the output directory. It links the PNGs and per-case
`result.json` and `blender.log`. `summary.json` includes source hashes, complete
commands, selected device IDs, effective settings, process exit codes, render-job
time and alpha-pixel counts. Scene-loading time is excluded from the reported
render-job time; first-use compilation/cache state is not standardized.

Render PASS means a completed GUI job, expected GPU/settings evidence, a decodable
PNG of the requested dimensions and an unchanged source file. It is **not** an
image-match verdict. EEVEE can differ from Cycles because it uses another rendering
method. Retain the original PNG alpha channel when checking compositor/edge issues.

Press Ctrl+C to stop the current owned Blender process, or create an empty file
named `STOP` in the printed output directory. The runner marks the run interrupted
and does not continue to the remaining cases. It does not terminate other Blender
processes.

## Compare images from two machines

Copy both PNGs to the same machine. Match scene hash, camera/frame, dimensions,
sampling, denoiser, bounce limits and color/compositor settings first.

```powershell
python compare_images.py .\reference.png .\native.png `
  --tolerance 16 --json .\results\image_metrics.json `
  --diff .\results\image_difference_x8.png
```

This uses the same formulas as the offline viewer `tools/image_compare.html`:
RGB MAE/RMSE, RGB PSNR, luma SSIM in 8×8 blocks, changed pixels above tolerance,
and separate alpha error. Images are not resized. Comparison exit 0 means metrics
were computed; differences do not trigger a failure threshold. See
[Image Compare details](../tools/IMAGE_COMPARE_README.md).

## Code checks

```powershell
python -m unittest discover -s tests -p test_native_render.py
python -m unittest discover -s tests -p test_image_compare_cli.py
```

These checks run without Blender/GPU execution. The native worker uses only
Blender's bundled Python and standard library; the host requires NumPy and Pillow.
