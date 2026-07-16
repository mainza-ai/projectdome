# Project Dome — Wiki Log

## [2026-07-15] ingest | Project Dome Free Tools Research
Ingested the comprehensive research doc covering state-of-the-art tooling for GNM-based conversational avatars — LLMs (Qwen3, DeepSeek-V4-Flash), TTS (Kokoro, Piper), forced alignment (Wav2TextGrid, MFA), deterministic and neural mapping approaches, WebGPU rendering.

## [2026-07-15] ingest | PROJECT_DOME_ROADMAP
Ingested the phased roadmap and data acquisition plan. Establishes Path A (lookup-table MVP) → Path B (neural regressor) evolution, dataset candidates, licensing reality check, and the $0 budget constraint.

## [2026-07-15] wiki | Initial wiki created
Bootstrapped wiki from dev-docs: index, overview, architecture, all layer pages, tool/dataset/license references, glossary, and roadmap.

## [2026-07-15] wiki | Codebase documentation added
Added wiki pages for every source module: server, mind-llm, voice-tts, alignment-pipeline, animation-engine, training-pipeline, reprojection, web-renderer, web-ui, gnm-sanity-check, export-tools, and setup. Updated index and log.

## [2026-07-15] wiki | Production audit added
Comprehensive gap analysis against google/GNM (8 gaps) and TimoBolkart/voca (6 gaps) with phased 6-phase implementation plan covering core architecture, full GNM expression control, rig kinematics, identity system, performance/latency, and testing.

## [2026-07-15] exec | Production audit execution (batch 1)
Completed Phase 7 (server hardening), Phase 1 (core architecture), Phase 2.1/2.4, Phase 3.1/3.3/3.4, and Phase 6. Details:
- Fixed Keras loading warnings via _SilentExpressionSampler with compile=False
- Structured logging, error boundaries, gc.collect memory guards in server
- VOCA-compatible 8/2/2 speaker-specific train/val/test split (critical data leakage fix)
- Edge loss added to training with gradient clipping
- Speaker ID embedded in reprojected NPZ files
- Enhanced emotion mappings (FEAR, ANGRY) and expanded web renderer with body-part vertex colors, independent eye gaze, natural blink timing distribution
- Pose correctives export in export_basis.py
- 5 unit test suites (viseme_mapper, viseme_table, emotion_blender, interpolator, dataset_split)
## [2026-07-15] exec | Production audit execution (batch 2)
Completed remaining phases: tongue animation coefficients, identity PCA slider bank UI, streaming audio delivery, and README screenshot.
- Tongue coefficients set for TH, DD, CH, kk, SS, RR, EE, OO, schwa visemes (32 tongue dimensions)
- /api/identity/info endpoint + 10 PCA component sliders in UI with -3 to 3 range
- /api/speak/stream endpoint splits text into sentences, processes each chunk, returns combined + per-chunk data
- Web client auto-selects streaming endpoint for text > 80 chars
- Screenshot added to README

## [2026-07-15] audit | Comprehensive codebase audit
## [2026-07-15] exec | Remaining gaps implemented
A1-A7 and B3-B6 completed and verified:
- A1: Buffer export added to setup.sh (Step 7)
- A2: 45-second request timeout via signal.alarm on all synthesis endpoints
- A3: evaluate.py speaker_ids fix (was already correct, verified)
- A4/B3: Reprojection skip-if-exists (partial progress saving)
- A5: Mind layer wired as /api/chat and /api/chat/reset — returns response_text + emotion + audio + visemes
- A6: Web client shows actionable error when buffers are missing ("run python tools/export_basis.py")
- B5: Training pipeline orchestrator (src/training/run_pipeline.py)
- B6: Structured logging for train.py, evaluate.py, reproject_vocaset.py
All 25 tests pass, all 9 API endpoints verified.

## [2026-07-15] fix | Mouth animation critical bug fix
Found and fixed two root causes of the mouth-not-moving bug:
1. **Web renderer bug**: `finalCoeffs` was overwriting blended speech coefficients with emotion upper-face values (line 352-353 in renderer.js). Fix: use `blended` directly.
2. **Wrong viseme coefficients**: Assumed PCA components corresponded to anatomical jaw opening — but `lower_face_region_000` (set to 1.2 for "aa") actually raises the upper lip, not opens the jaw. Used GNM model probing to discover that `lower_face_region_001` (index 1 in 182-dim space) is the true jaw-opening component. Rebuilt entire viseme table with verified GNM-compatible coefficients. Verified: "aa" now produces lip center downward displacement of -0.011 units.

## [2026-07-15] audit | Deep dive into GNM and VOCA codebases
Full source-code audit of vendor/GNM/gnm/shape/ (gnm_xnp.py, gnm_common.py, gnm_numpy.py) and vendor/voca/ (run_voca.py, utils/). Identified 18 gaps across 4 categories.

## [2026-07-15] exec | All 18 gaps implemented
Complete implementation of all deep-dive audit findings:
- GNM export: pose_correctives_regressor, joint_identity_basis, vertex_groups (46 groups), mirror_indices, triangle_uvs — all exported correctly with proper attribute names
- Web renderer: per-body-part vertex colors using GNM's 46 group names, joint identity basis applied to skeleton, proper LBS skinning math with homogeneous coordinates, blink timing
- Server hardening: temp file cleanup in alignment (NamedTemporaryFile), LRU synthesis cache (64 entries), model download validation (file size check)
- Training: HuBERT/DeepSpeech feature extractor abstraction, vertex-space loss (projects coefficients through GNM), YAML config system with TrainingConfig dataclass
- Testing: 4 GNM forward pass regression tests (template, basis, forward, vertex count), 9 server integration tests (all endpoints + static files), 30 total tests passing
- pose_correctives_regressor verified all zeros in v3.0 HEAD model — exported only if non-zero

## [2026-07-15] fix | Emotion blend rewrite, head centering, Conv1D model
- Emotion blend now uses speech-energy gate: when speaking, lower face = speech ONLY (emotion only in upper face). When silent, lower face = emotion fully. Zero interference between speech and emotion.
- Camera repositioned: 40° FOV, centered on head at y=0.24, controls target z=0.03
- Added VOCA-style Conv1D frontend option to SpeechToCoefficientsModel (4-layer Conv1D with stride-2, BatchNorm, configurable size factor)
- All 31 unit tests + 9 server integration tests passing

## [2026-07-15] fix | Head centering — bounding sphere camera framing
Replaced hardcoded camera position with computed framing using THREE.Sphere bounding sphere: camera looks at bsphere.center, distance = radius * 1.5 / sin(FOV/2). Added diagnostic console.log for head bounds, camera position, and distance. Removed CSS !important canvas sizing rules that conflicted with Three.js setSize. Head bounds verified from Python: center=(0, 0.254, 0.067), radius=0.233.

## [2026-07-15] fix | Emotion/speech blend conflict
CVAE emotion coefficients span the entire lower face (up to 4.7 magnitude for BLOW), conflicting with speech visemes. Fixed blend to use per-channel magnitude comparison: where speech energy > 0.15*emotion energy, speech wins; otherwise emotion tints the lower face subtly. Upper face (0-199) and pupils remain full emotion. Verified: 26 tests pass, all endpoints working.

Full audit after production execution: 12 items fixed, 10 remaining gaps identified (A1-A7 critical, B1-B6 quality, C1-C4 performance), 5 missing features documented. Key fixes: eager import chains, Keras warnings fully suppressed, all 25 tests passing, server starts cleanly.

## [2026-07-15] fix | LBS skinning double-invBind bug + wiki troubleshooting section
Found and fixed the root cause of "head stuck at bottom of viewport": the LBS skinning matrix formula in `deformMesh()` was computing `T = makeTranslation(jw - R*jr)` instead of `T = makeTranslation(jw)`. The extra `-R*jr` subtraction doubled with `invBind`'s `-jr` to produce `-2*R*jr`, shifting every vertex by `-sum(w_j × joint_position_j)` — approximately −0.207 in Y. Camera was positioned correctly for the unshifted template; the skinning was the silent conflict.

Created troubleshooting.md with 10 documented issues covering: LBS skinning bug, camera distance, buffer loading failures, mouth animation, missing triangles, OrbitControls jumping, emotion blend weakness, audio/viseme sync, server startup, and integration test setup. Updated web-renderer.md with current camera positioning and LBS details.

## [2026-07-15] audit | VOCA vs GNM architecture deep-dive
Completed a 22-file deep-dive audit comparing VOCA's speech animation architecture against GNM's model parameterization and our current integration. 7 key findings documented:
1. VOCA outputs raw vertex offsets (3×V), not model parameters — bypasses PCA entirely
2. GNM's expression PCA maxes at 0.008 displacement per unit coefficient — ~100× too weak for speech
3. GNM has no jaw joint (4 joints: neck, head, eyes); FLAME has explicit jaw rotation (pose[6:9])
4. VOCA's ExpressionLayer initializes from FLAME's basis, then trains — our model should follow the same pattern
5. VOCA's training loss operates on vertices, not coefficients — our pipeline should too
6. Correct separation: speech uses vertex offsets, emotion uses PCA (upper face only), no blend function needed
7. Current code makes 7 specific architectural mistakes (documented in voca-gnm-audit.md)

## [2026-07-15] audit + plan | Full speech animation architecture audit
Completed a 18-file deep-dive audit of the entire speech animation pipeline (GNM library, server, alignment, web client, training model). Identified 8 architectural gaps preventing production-quality mouth animation:
1. No jaw joint in skeleton — PCA displacement max 0.008/unit (vs 0.018 from 5.7° joint rotation)
2. Binary speech-energy gate removes emotion contribution
3. Pose correctives all zeros (model data limitation)
4. PCA coefficient scale too small for speech (needs 10-20×)
5. Web LBS double-invBind bug (fixed)
6. No identity-calibrated viseme scaling
7. No alignment pipeline fallback
8. No GPU compute for deformation

Created wiki/speech-production-plan.md with P0-P3 prioritized implementation plan. Updated README with VOCA download instructions. Updated wiki/index.md and roadmap.md to reference the new plan.
