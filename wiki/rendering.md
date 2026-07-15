# Rendering Layer

Deploys the 17,821-vertex GNM mesh with 383 morph targets at 60fps in a web browser.

## Technology stack

- **WebGPU** — modern low-overhead graphics API (supersedes WebGL)
- **Three.js WebGPURenderer** — high-level 3D library with WebGPU backend
- **Three Shading Language (TSL)** — JS/TS-based node system that compiles to WGSL

## The 383-morph-target problem

WebGL limits meshes to 4–8 active morph targets per frame. GNM frequently needs dozens of non-zero expression coefficients simultaneously. The solution: **pack the entire identity and expression basis into a DataTexture** and compute deformation in a custom vertex shader.

## TSL vertex deformation pipeline

```
CPU: resolve final 383-dim expression + 253-dim identity vectors
  → pass to GPU as Uniform/Storage Buffer

GPU (TSL vertex shader):
  1. Retrieve base vertex position from template mesh
  2. Loop over active coefficients
  3. Fetch basis vectors from DataTexture (indexed by vertex ID / UV)
  4. final_position = template + identity_basis·id + expression_basis·expr
  5. Output final vertex
```

**Result:** ~41 MFLOPs of dense matrix multiplication per frame offloaded entirely to the GPU ALUs. The CPU is free to run LLM inference, TTS, and forced alignment concurrently without dropping frames.

## Deployment targets

| Target | Priority | Tech |
|---|---|---|
| Browser | Primary | WebGPU / Three.js / TSL |
| Native | Secondary | RealityKit (MaiSOUL integration) |

## Prototyping

Start in Blender's real-time viewport with GNM Head Importer before committing to a web renderer.

## Implementation

See [[web-renderer|Web Renderer]] for the current Three.js implementation: CPU-side buffer loading, sparse deformation, LBS, blink simulation, and render loop.
