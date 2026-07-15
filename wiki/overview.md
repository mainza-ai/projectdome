# Project Overview

**Project Dome** is a **Generative Anthropomorphic Model (GNM)-based conversational 3D avatar engine** — a fully local, real-time talking head that runs on a $0 budget with no metered cloud services.

## Core idea

A conversational 3D avatar with four layers:

1. **Mind** — a local LLM generates dialogue and semantic affect tags
2. **Voice** — a local neural TTS engine produces synchronized audio
3. **Mapping** — audio and emotion tags drive GNM's 383-dimensional expression space
4. **Renderer** — WebGPU/Three.js renders the 17,821-vertex mesh in real-time

## Key constraints

- **$0 budget** — no paid APIs, no metered services, no credit card required
- **Fully local** — all inference runs on consumer hardware (CPU/GPU)
- **Permissive licensing** — Apache 2.0 where possible, no GPL or research-only dependencies in production
- **Browser-first** — WebGPU deployment, native secondary

## Architecture

See [[architecture|Architecture]] for the full pipeline.

## Roadmap

See [[roadmap|Development Roadmap]] for the phased build plan.
