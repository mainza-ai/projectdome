# Project Dome — Roadmap & Data Acquisition Plan

*A GNM-based conversational 3D avatar engine ("the arbiter")*

Status: pre-build. Only asset in hand: `google/GNM` (Apache 2.0), verified as a linear 3DMM — 17,821 vertices, 253 identity params, 383 expression params, 4-joint skeleton, built-in 20-label `ExpressionSampler` CVAE, no speech mapping.

**Budget constraint: $0.** Every tool and dataset below was chosen (or re-chosen) to be free with no metered usage — no pay-per-character TTS, no paid API tiers you could accidentally exceed. Where a free tier of a paid cloud service was tempting, it's swapped for a fully open-source local alternative instead, so there's no cap to hit and no billing surface at all.

Initialize the GitHub repository https://github.com/mainza-ai/projectdome.git
Clone https://github.com/google/GNM/tree/main ensure no GitHub conflicts with main repo https://github.com/mainza-ai/projectdome

---

## 1. What you actually need to source

Project Dome has four missing layers. Only **one** of them needs a real dataset. The other three need APIs, tooling, or nothing at all. Knowing which is which keeps you from over-collecting.

| Layer | Needs a dataset? | Free option |
|---|---|---|
| Mind (LLM response + intent) | No | An open-weight local model (run via Ollama or similar) avoids per-token cost entirely; if you use a hosted API instead, watch its free-tier limits carefully since those do reset/cap |
| Voice (TTS + timing) | No | Piper TTS (local, MIT license) + Montreal Forced Aligner (local, open source) — no usage cap, ever |
| **Mapping (audio/emotion → GNM coefficients)** | **Yes** | This is the unsolved part — no pretrained GNM mapper exists anywhere. Free datasets exist (Section 3), just under research-only licenses |
| Renderer | No | WebGPU/WebGL2 (browser-native, free), Blender (free, open source), RealityKit (free with Apple developer tools) |

So the entire dataset-acquisition problem reduces to one question: **what do you train the mapping model on?** Below are your real options, ranked by effort.

---

## 2. Path A (fastest — no 3D dataset required, zero cost): local TTS + free forced alignment

Skip cloud TTS entirely — the free tiers on Polly/Azure/etc. are capped per month and are a metered service sitting under your project, which isn't what you want at $0 budget. Instead, run everything locally with open-source tools that have no usage limit at all:

- **Piper TTS** (`OHF-Voice/piper1-gpl`, MIT-licensed) — a fast, fully local neural TTS engine. Installs with `pip install piper-tts`, runs on CPU in real time (no GPU required), ships 100+ downloadable voices across 30+ languages, and embeds `espeak-ng` for phonemization. No API key, no internet call at inference time, no per-character cost, ever.
- **Montreal Forced Aligner (MFA)** (open source, Kaldi-based) — takes any `(audio, transcript)` pair and produces precise phone-level and word-level timestamps (as Praat TextGrid files). Run this on whatever Piper generates and you get exactly the same kind of "audio → timestamped phoneme" data that Polly's viseme marks would have given you, except it's free, offline, and unlimited. MFA ships pretrained English acoustic models and a G2P model so you don't need your own pronunciation dictionary.

**Pipeline:** text → Piper (audio + phoneme sequence from its embedded espeak-ng) → MFA forced-alignment (phoneme sequence + audio → millisecond-accurate phoneme timestamps) → your own small phoneme→viseme-category lookup (a standard, well-documented mapping — English phonemes collapse into roughly a dozen visually-distinct mouth-shape categories) → GNM coefficient lookup.

**What this buys you:** a free, infinite, precisely-timestamped `(audio, viseme-category, timestamp)` stream for any text, in any locally-installed voice. No one's face needs to be recorded, no cloud account needs to be created, and nothing here has a monthly quota to run out of.

**What it doesn't give you:** GNM coefficients directly. You bridge this gap by hand-authoring (once, manually, in the free Blender GNM Head Importer add-on — Blender itself is free and open source) a reference pose for each of the ~12–15 viseme categories your phoneme mapping produces, expressed as coordinates in GNM's `lower_face_region_*` + `tongue_*` basis. Then the "model" is just: MFA emits timestamped viseme categories → look up the corresponding hand-authored GNM coefficient vector → interpolate between them over time. This is not a trained neural network — it's a lookup table plus interpolation — but it's a complete, real-time, shippable MVP lip-sync pipeline, buildable in days, and it costs nothing to run at any scale.

**When to use this path:** for your first working demo, and possibly permanently for a v1 product. Classic viseme-driven rigs (this is literally how most game and VTuber lip-sync worked for a decade) are proof this approach produces convincing results without any ML training or any paid service at all.

---

## 3. Path B (higher fidelity — needs 3D datasets + re-projection): trained regressor

If Path A's ~15-category lookup table feels too coarse (it won't capture coarticulation, subtle jaw/tongue dynamics, or emotional coloring of speech), the next step is a real regressor trained on continuous audio → continuous mesh-deformation data, then re-projected onto GNM's basis.

None of the datasets below are natively in GNM's coordinate space — that re-projection step (Section 4) is the actual novel work, and it's exactly what `fitting_utils/project_on_pca.py` in the GNM repo is for.

### 3.1 Primary candidates (speech-to-3D-face datasets, ranked by usefulness)

**VOCASET** (MPI-IS) — <cite index="2-1">roughly 29 minutes of 4D scans at 60 fps with synchronized audio from 12 speakers</cite>, meshes registered to the FLAME topology. Because FLAME and GNM are both parametric head models trained on real scan data, FLAME→GNM re-projection is the most tractable of the available options. Free for non-commercial research use; requires registering an account at the MPI-IS VOCA site. Best starting point.

**BIWI 3D Audiovisual Corpus** — <cite index="4-1">40 unique sentences spoken by 14 subjects (8 female, 6 male), each sentence spoken once emotionally and once neutrally, captured at 25 fps with 23,370 vertices per frame</cite>. Good for emotional-speech variation since VOCASET is largely neutral-toned; topology differs from FLAME/GNM so it needs its own re-projection (raw scan geometry, not a shared parametric basis).

**Multiface** (Meta Reality Labs Research) — <cite index="18-1">a multi-view dataset of 13 identities performing facial expressions, roughly 40 to 160 camera views per frame with tracked meshes, audio, and calibration metadata, totaling around 65TB</cite>. Far higher fidelity than VOCASET/BIWI, but enormous in size and multi-view-camera-oriented rather than clean 4D-scan-oriented — most useful later, for refining fine expression detail, not as a first dataset. Open-sourced by Meta with code to download and build a Codec Avatar baseline.

**MEAD** — <cite index="33-1">a talking-face video corpus of 60 actors and actresses performing 8 emotions at 3 intensity levels each, captured from 7 viewing angles</cite>. This is 2D video, not 3D geometry, so it can't feed the mesh regressor directly — its real value is as an *emotion-in-speech* reference: you can run a 2D face-landmark or ARKit-blendshape extractor over it to get an emotion-conditioned expression signal, then use that as auxiliary supervision or to validate your emotion-to-coefficient mapping.

### 3.2 Why re-projection is required either way

None of these datasets ship in GNM's basis, because GNM is days old and nothing has been fit to it publicly yet. The bridging step is always: **target dataset mesh → GNM's `project_on_pca.py` fitting tool → GNM coefficients**, then train audio-to-coefficient regression on the *output* of that projection, not on the raw dataset. This means dataset quality matters less than topology compatibility — VOCASET/FLAME is the easiest re-projection target because both are linear PCA-style scan-based models; BIWI and Multiface are raw high-res scans and will need a heavier fitting step (ICP-style mesh alignment before PCA projection).

### 3.3 Licensing reality check (and cost check)

All four datasets above are **free to download** — no purchase, no subscription. VOCASET requires registering an account at the MPI-IS site; BIWI, Multiface, and MEAD are downloadable via their GitHub/project pages. The only real constraint is usage rights, not money: all four are **research/non-commercial licenses by default**. If Project Dome is headed toward a commercial product later, budget time to either (a) contact the dataset owners about commercial licensing, (b) rely on Path A's hand-authored data plus your own recorded data (you own full rights to anything you personally capture, at zero cost), or (c) generate a synthetic dataset by rendering a GNM-fitted face reciting phonetically-balanced sentences via Piper TTS and running a free/open audio-to-blendshape model, then re-projecting that output onto GNM — sidestepping third-party dataset licensing entirely.

**Compute for training, if you get to Phase 4:** you don't need to buy GPU time either. Google Colab and Kaggle Notebooks both offer free GPU hours (with session-length and weekly-quota limits), which is enough for a VOCASET-scale regressor — these datasets are small (tens of minutes to a few hours of data), not the kind of thing that needs a paid cluster.

---

## 4. Recommended roadmap, phased

**Phase 0 — Verify the environment (you've mostly done this already)**
Confirm mesh size, basis linearity, expression channel names, `ExpressionSampler` label set. Done.

**Phase 1 — Ship a lookup-table MVP (Path A)**
- Install Piper TTS locally and Montreal Forced Aligner; confirm you can generate audio for arbitrary text and get back phoneme-level timestamps for free, offline.
- Map English phonemes down to ~12–15 visual viseme categories (a standard, well-documented reduction — many public phoneme-to-viseme tables exist).
- In Blender (free, GNM Head Importer add-on), hand-pose GNM's expression sliders for each viseme category, export each as a coefficient vector.
- Build the runtime: MFA-derived viseme timeline → coefficient lookup + linear interpolation → feed into GNM's `final_vertices = template + identity_basis·id + expression_basis·expr` skinning math → render.
- Layer in the built-in `ExpressionSampler`'s 20 semantic emotion labels (HAPPY, SURPRISE, etc.) for non-speech affect, blended additively with the viseme signal.
- This alone is a working, demoable "talk to it and it talks back with 3D expression" pipeline.

**Phase 2 — Pick a rendering target and get it real-time**
- Browser: WebGPU compute shader (preferred) or WebGL2 vertex shader with the basis packed into a data texture, per your own earlier MFLOPs analysis.
- Native: RealityKit path for MaiSOUL integration, feeding GNM's numpy/PyTorch output into vertex buffers.
- Prototype fastest in Blender's real-time viewport before committing to either.

**Phase 3 — Start the re-projection pipeline (Path B groundwork)**
- Download VOCASET (register at MPI-IS), run `fitting_utils/project_on_pca.py` against it to produce a first GNM-basis speech dataset.
- Validate: does re-projected VOCASET, driven through the same lookup+interpolation runtime from Phase 1, look smoother/more accurate than the hand-authored table?

**Phase 4 — Train the real regressor**
- Once re-projected VOCASET data exists, train a small sequence model (phoneme/audio-feature input → 182-dim lower-face+tongue coefficient output) — a lightweight Transformer or even an LSTM is plausible given VOCASET's modest size; look at how VOCA/FaceFormer/CodeTalker structured this problem, since they solved the same input/output shape for FLAME.
- Fold in BIWI (emotional variant) once the neutral-speech case works, to get emotional coloring into the same model instead of only in the separate `ExpressionSampler`.

**Phase 5 — Scale and polish**
- Bring in Multiface for fine detail once the core pipeline works, if photorealism (not just correctness) becomes the goal.
- Revisit licensing before any commercial launch.

---

## 5. Instruction prompt — hand this to an AI coding agent to bootstrap the repo

Copy everything in the box below into a fresh session with a coding-capable AI (Claude Code, etc.) once you're ready to start building:

```
You are bootstrapping "Project Dome" — a from-scratch conversational 3D avatar
engine built on Google's GNM Head model (github.com/google/GNM, Apache 2.0).

HARD CONSTRAINT: This project has a $0 budget. Do not integrate any paid API,
any metered cloud service, or anything that requires a credit card, even if
it has a "free tier." Prefer local/open-source tools with no usage cap. If a
step genuinely requires a paid service and there is no free alternative, stop
and tell me instead of proceeding.

CONTEXT (already verified, do not re-derive):
- GNM Head is a linear 3DMM: 17,821 vertices, 253 identity params, 383
  expression params (grouped as left_eye_region_000-099, right_eye_region_000-099,
  lower_face_region_000-149, tongue_mean + tongue_000-030, pupils_000), 4-joint
  skeleton, linear blend skinning + pose correctives. ~41 MFLOPs per frame for
  expression deformation.
- There is a built-in ExpressionSampler (CVAE, expression_decoder_model.h5) that
  maps 20 semantic labels (SURPRISE, DISGUST, SUCK, COMPRESS_FACE, STRETCH_FACE,
  HAPPY, SQUINT, PLATYSMA, BLOW, FUNNELER, SMILE_WIDE, CORNERS_DOWN, PUCKER,
  WINK_LEFT, WINK_RIGHT, MOUTH_LEFT, MOUTH_RIGHT, LIPS_ROLL_IN, SNARL,
  TONGUE_CENTER) to expression coefficients. No speech/viseme mapping exists
  anywhere in the repo.
- `fitting_utils/project_on_pca.py` exists in the repo and can project external
  target geometry onto GNM's PCA basis.

YOUR TASK, IN ORDER:

1. Clone google/GNM and set up the numpy or PyTorch backend. Confirm you can
   load the v3_0 gnm_head.npz, generate a neutral mesh, and apply a test
   expression coefficient vector to visibly deform it (dump to .obj or render
   in a headless viewer, whichever is easier to sanity-check).

2. Build a "Path A" MVP first, before any ML training:
   a. Install Piper TTS (pip install piper-tts, MIT license, fully local/free)
      and Montreal Forced Aligner (open source, local). Build a pipeline:
      text -> Piper audio -> MFA forced alignment against the same text ->
      phoneme-level timestamps. Reduce phonemes to ~12-15 visual viseme
      categories using a standard phoneme-to-viseme table.
   b. Hand-author (or accept placeholder/manually-tuned) GNM coefficient
      vectors for each viseme category the TTS emits, targeting the
      lower_face_region_* and tongue_* channels.
   c. Build a runtime loop: viseme timeline -> coefficient lookup with linear
      interpolation between keyframes -> feed into GNM's skinning math each
      frame -> output a vertex buffer.
   d. Wire the 20-label ExpressionSampler in additively for non-speech affect
      (e.g. an LLM-tagged emotion like "HAPPY" blends with whatever the mouth
      is currently doing for speech).

3. Build the smallest possible renderer to prove real-time performance:
   prefer a WebGPU compute-shader or WebGL2 vertex-shader prototype in-browser
   (pack the identity/expression basis into a data texture), since browser
   deployment is the priority target. Native RealityKit integration comes later.

4. Only after step 2-3 work end to end, start on the higher-fidelity "Path B"
   regressor:
   a. Help me register for and download VOCASET (MPI-IS, non-commercial
      research license) as the first re-projection source, since it shares
      FLAME's parametric-scan lineage with GNM.
   b. Use fitting_utils/project_on_pca.py to re-project VOCASET's FLAME-topology
      sequences into GNM's basis. Produce a dataset of (audio, GNM-coefficient
      sequence) pairs.
   c. Propose and scaffold a small sequence model (phoneme/audio-feature input
      -> 182-dim lower_face+tongue coefficient output over time) trained on
      that re-projected data. Look to VOCA / FaceFormer / CodeTalker's
      published architectures for the input/output framing, since they solved
      the equivalent problem for FLAME.
   d. Leave BIWI (emotional-speech 3D scans, non-FLAME topology) and Multiface
      (Meta, high-fidelity multi-view, ~65TB) as later-phase datasets for
      emotional coloring and fine detail respectively — flag their licensing
      as research-only and remind me to revisit before any commercial launch.

5. At every stage, keep the "mind" (LLM) and "voice" (TTS) layers swappable
   behind clean interfaces — don't hardcode to one LLM or TTS provider, since
   those are commodity layers, not part of the actual novel engineering
   surface of this project. For the "mind" layer, default to a free/local
   open-weight model unless I tell you otherwise.

Ask me before downloading any dataset over a few GB, before making any
architecture decision that would lock us into a specific renderer (browser vs.
native) earlier than necessary, and before integrating any service that isn't
clearly free with no usage cap.
```

---

## 6. Quick reference — where everything lives

| Resource | URL / access path | License |
|---|---|---|
| GNM (base model) | github.com/google/GNM | Apache 2.0 |
| VOCASET | voca.is.tue.mpg.de (register required) | Non-commercial research |
| BIWI | ETH Zurich CVL, via original paper's data request process | Research use |
| Multiface | github.com/facebookresearch/multiface | Meta research license, ~65TB |
| MEAD | github.com/uniBruce/Mead | Research use |
| Amazon Polly viseme API | AWS Polly `SynthesizeSpeech`, `SpeechMarkTypes: ["viseme"]` (Standard/Neural engines only) | Commercial, pay-per-character |
| Azure viseme API | Azure AI Speech, `VisemeReceived` event | Commercial, pay-per-character |
| Blender GNM Head Importer | Community add-on for Blender 5.1+ | Check add-on's own license |

Note: exact dataset access URLs and terms can shift — confirm current license terms directly on each site before downloading, especially if Project Dome moves toward commercial release.
