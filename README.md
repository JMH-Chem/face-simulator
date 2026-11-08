# Face Simulator

A real-time, photoreal digital human head that runs in the browser (Three.js / WebGL 2).
The person is synthetic: the face shape is sampled from a statistical model built from
light-stage scans, so it does not depict any real individual.

## What is in it

| Area | Technique |
| --- | --- |
| Shape | ICT-FaceKit morphable model: 40 identity PCA modes + 53 ARKit-style expression blend shapes, blended on the CPU with welded normals |
| Skin colour | Two-chromophore model (melanin + hemoglobin, Beer-Lambert absorption) so tone and flushing stay adjustable |
| Skin light transport | Screen-space separable subsurface scattering with the d'Eon/Luebke six-Gaussian skin profile, applied to irradiance only so pigment detail stays sharp; thin-part transmission (ears, nostrils) gated by the shadow map |
| Skin surface | Dual-lobe specular (base + oily clear coat by region), tileable micro-relief with pores, macro age lines, expression-driven wrinkle maps (brow raise, smile/squint, frown/sneer) |
| Eyes | Custom eyeball with corneal bulge; view ray refracted (n = 1.376) onto a recessed iris plane, adjustable pupil, procedural iris fibres/crypts and scleral vessels, tear film and lid occlusion |
| Lighting | Procedural photo-studio rigs prefiltered into image-based lighting (softboxes appear as catch-lights) plus a key light with PCSS contact-hardening soft shadows |
| Camera | Thin-lens depth of field from real focal length / f-number, bloom, Khronos PBR Neutral tone mapping, grain, vignette |
| Behaviour | Blinks, saccades, lid–gaze coupling, breathing, idle head motion, expression presets, babble talking, cursor gaze |

## Run it

Any static file server works (the page fetches its assets with `fetch`):

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000/.

## Rebuild the assets

The generated data lives in `assets/` and is committed. To regenerate it:

```bash
git clone --depth 1 https://github.com/ICT-VGL/ICT-FaceKit /tmp/ict
pip install numpy scipy pillow trimesh embreex
python3 tools/build_model.py --ict /tmp/ict --out assets
python3 tools/bake_geometry.py --cache /tmp/facecache
python3 tools/build_textures.py --cache /tmp/facecache
python3 tools/build_detail_textures.py
```

| Script | Output |
| --- | --- |
| `tools/build_model.py` | `assets/face.json`, `assets/face.bin.gz` (topology, UVs, quantised identity/expression deltas) |
| `tools/bake_geometry.py` | `assets/face_extra.bin` (ray-traced per-vertex AO and thickness) + per-texel geometry cache |
| `tools/build_textures.py` | skin maps in `assets/tex/` (pigment, normals, roughness/pores, stubble, wrinkles) |
| `tools/build_detail_textures.py` | pore tile, iris, sclera, eyelashes |

## Credits

Face shape and expression data: [ICT-FaceKit](https://github.com/ICT-VGL/ICT-FaceKit),
USC Institute for Creative Technologies, MIT License (see `licenses/ICT-FaceKit-LICENSE.txt`).
All textures, shaders and lighting are generated procedurally by the scripts in this repository.
