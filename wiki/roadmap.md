For the detailed gap analysis and prioritized implementation plan, see the [[speech-production-plan|Speech Animation Production Plan]].

## Pre-Phase 0 — Repository Bootstrap
- [ ] Initialize git and remote `mainza-ai/projectdome`
- [ ] Add `google/GNM` as a git submodule in `vendor/GNM/`
- [ ] Configure `.gitignore` to protect large VOCASET files and compile artifacts
- [ ] Push bootstrap commit to GitHub

## Phase 0 — GNM Foundation & Verification
- [ ] Create Python 3.13 venv, do editable install of GNM shape package from submodule
- [ ] Run sanity check: load v3_0 `gnm_head.npz`, generate neutral mesh, apply test expression (`HAPPY` vector via `ExpressionSampler`), and export OBJ meshes to verify displacement.

## Phase 1 — Lookup-table MVP (Path A)
- [ ] Install Piper TTS + Montreal Forced Aligner locally
- [ ] Map English phonemes → ~12–15 viseme categories
- [ ] Hand-pose GNM expression sliders in Blender for each viseme
- [ ] Build runtime: viseme timeline → coefficient lookup + interpolation → skinning math
- [ ] Layer in ExpressionSampler for non-speech affect
- **Deliverable:** working "talk to it and it talks back" 3D avatar

## Phase 2 — Real-time rendering
- [ ] Prototype in Blender viewport
- [ ] Build WebGPU/Three.js renderer with DataTexture-based deformation
- [ ] Alternative: RealityKit for native integration
- **Deliverable:** real-time browser-based renderer at 60fps

## Phase 3 — Re-projection pipeline (Path B groundwork)
- [x] Verify VOCASET data (~16.6 GB) and pre-trained VOCA models exist locally at `voca/` (Done)
- [ ] Run `fitting_utils/project_on_pca.py` to project FLAME-topology sequences onto GNM expression basis
- [ ] Extract 182-dim lower face + tongue coefficients from the projection results
- [ ] Pair GNM coefficient sequences with audio features
- **Deliverable:** first `(audio, GNM_coefficient)` training pairs on GNM's basis

## Phase 4 — Train neural regressor
- [ ] Train small sequence model (HuBERT → 182-dim lower face + tongue output)
- [ ] Fold in BIWI for emotional variation
- **Deliverable:** learned lip-sync with coarticulation

## Phase 5 — Scale and polish
- [ ] Multiface for fine detail (if photorealism is the goal)
- [ ] Revisit licensing before commercial launch
- **Deliverable:** production-ready avatar engine
