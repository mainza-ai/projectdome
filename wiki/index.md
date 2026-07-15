# Project Dome Wiki — Index

*Last updated: 2026-07-15*

A GNM-based conversational 3D avatar engine ("the arbiter"). $0 budget, fully local/open-source infrastructure.

## Overview

- [[overview|Project Overview]] — what, why, and how
- [[architecture|Architecture]] — the 4-layer pipeline
- [[roadmap|Development Roadmap]] — phased build plan
- [[setup|Setup & Environment]] — getting started, project structure
- [[glossary|Glossary]] — key terms

## Core Model

- [[gnm-head|GNM Head]] — Google's parametric 3DMM (17,821 vertices, 383 expression params)
- [[gnm-sanity-check|GNM Sanity Check]] — model loading verification

## Architecture Layers

- [[cognitive-layer|Cognitive Layer (Mind)]] — LLM orchestration (design overview)
- [[mind-llm|Mind LLM Engine]] — provider protocol, LocalMindProvider, conversation context
- [[acoustic-layer|Acoustic Layer (Voice)]] — TTS design comparison
- [[voice-tts|Voice TTS Provider]] — Piper TTS implementation
- [[temporal-alignment|Temporal Alignment]] — phonetic forced alignment (concept)
- [[alignment-pipeline|Alignment Pipeline]] — Wav2TextGrid + viseme mapper implementation
- [[animation-engine|Animation Engine]] — viseme table, interpolator, emotion blender, GNM driver, runtime loop
- [[mapping-path-a|Path A — Deterministic Viseme Mapping]] — lookup-table MVP
- [[mapping-path-b|Path B — Neural Regression]] — FaceDiffuser-inspired regressor
- [[web-renderer|Web Renderer]] — Three.js mesh deformation, LBS, rendering loop
- [[web-ui|Web UI]] — HTML, CSS, audio sync, animation controller
- [[rendering|Rendering Layer]] — WebGPU / Three.js / TSL (concept)

## Training

- [[training-pipeline|Training Pipeline]] — model architecture, dataset, training loop, evaluation
- [[reprojection|VOCASET Reprojection]] — ICP + PCA reprojection to GNM space

## Production Readiness

- [[production-audit|Production Audit & Integration Plan]] — gap analysis vs GNM + VOCA, phased implementation plan

## Tools & Stack

- [[tools|Tool Stack Reference]] — all tools with licenses and roles
- [[export-tools|Export & Tuning Tools]] — basis exporter, viseme tuner CLI
- [[licensing|License Analysis]] — permissive vs. research-only licenses

## Data

- [[datasets|Datasets]] — VOCASET, BIWI, Multiface, MEAD

## Sources

- `dev-docs/Project Dome Free Tools Research.md` — state-of-the-art tooling and architectural blueprint
- `dev-docs/PROJECT_DOME_ROADMAP.md` — roadmap and data acquisition plan
- `voca/` — VOCA model weights and preprocessed training data
