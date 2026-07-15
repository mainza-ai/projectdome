# Project Dome Wiki — Index

*Last updated: 2026-07-15*

A GNM-based conversational 3D avatar engine ("the arbiter"). $0 budget, fully local/open-source infrastructure.

## Overview

- [[overview|Project Overview]] — what, why, and how
- [[implementation-plan|Implementation Plan]] — bootstrapping and engineering steps
- [[architecture|Architecture]] — the 4-layer pipeline
- [[roadmap|Development Roadmap]] — phased build plan
- [[glossary|Glossary]] — key terms

## Core Model

- [[gnm-head|GNM Head]] — Google's parametric 3DMM (17,821 vertices, 383 expression params)

## Architecture Layers

- [[cognitive-layer|Cognitive Layer (Mind)]] — LLM orchestration, intent tagging
- [[acoustic-layer|Acoustic Layer (Voice)]] — TTS synthesis
- [[temporal-alignment|Temporal Alignment]] — phonetic forced alignment
- [[mapping-path-a|Path A — Deterministic Viseme Mapping]] — lookup-table MVP
- [[mapping-path-b|Path B — Neural Regression]] — FaceDiffuser-inspired regressor
- [[rendering|Rendering Layer]] — WebGPU / Three.js / TSL

## Tools & Stack

- [[tools|Tool Stack Reference]] — all tools with licenses and roles
- [[licensing|License Analysis]] — permissive vs. research-only licenses

## Data

- [[datasets|Datasets]] — VOCASET, BIWI, Multiface, MEAD

## Sources

- `dev-docs/Project Dome Free Tools Research.md` — state-of-the-art tooling and architectural blueprint
- `dev-docs/PROJECT_DOME_ROADMAP.md` — roadmap and data acquisition plan
- `voca/` — VOCA model weights and preprocessed training data
