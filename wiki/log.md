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

Full audit after production execution: 12 items fixed, 10 remaining gaps identified (A1-A7 critical, B1-B6 quality, C1-C4 performance), 5 missing features documented. Key fixes: eager import chains, Keras warnings fully suppressed, all 25 tests passing, server starts cleanly.
