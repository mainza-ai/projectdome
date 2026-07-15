<!-- codebase-memory-mcp:start -->
# Codebase Knowledge Graph (codebase-memory-mcp)

This project uses codebase-memory-mcp to maintain a knowledge graph of the codebase.
ALWAYS prefer MCP graph tools over grep/glob/file-search for code discovery.

## Priority Order
1. `search_graph` — find functions, classes, routes, variables by pattern
2. `trace_path` — trace who calls a function or what it calls
3. `get_code_snippet` — read specific function/class source code
4. `query_graph` — run Cypher queries for complex patterns
5. `get_architecture` — high-level project summary

## When to fall back to grep/glob
- Searching for string literals, error messages, config values
- Searching non-code files (Dockerfiles, shell scripts, configs)
- When MCP tools return insufficient results

## Examples
- Find a handler: `search_graph(name_pattern=".*OrderHandler.*")`
- Who calls it: `trace_path(function_name="OrderHandler", direction="inbound")`
- Read source: `get_code_snippet(qualified_name="pkg/orders.OrderHandler")`
<!-- codebase-memory-mcp:end -->

<!-- projectdome-wiki:start -->
# Project Dome Wiki Schema

This project maintains a structured wiki at `wiki/` — a collection of interlinked markdown files that serve as a persistent, evolving knowledge base. The LLM owns this layer entirely: creating pages, updating them when new information arrives, maintaining cross-references, and keeping everything consistent.

## Wiki conventions

- **Interlinking** — pages reference each other via `[[page-name|display text]]` wiki-style links or standard markdown `[text](page.md)` links. The wiki should form a connected graph — no orphan pages.
- **Frontmatter** — every page should have YAML frontmatter: `tags`, `date` (last updated), and `source_count` where applicable.
- **Bidirectional linking** — when creating or updating a page, also update any pages that reference it and the index.

## Operations

### Ingest a new source
1. Read the source document (raw sources live in `dev-docs/`)
2. Discuss key takeaways with the user
3. Write or update relevant wiki pages (entity pages, concept pages, the overview)
4. Update `wiki/index.md` — add new pages to the catalog
5. Append an entry to `wiki/log.md` with prefix `## [YYYY-MM-DD] ingest | Title`

### Answer a query
1. Read `wiki/index.md` to find relevant pages
2. Read the identified pages
3. Synthesize an answer with citations
4. If the answer is substantial (analysis, comparison, discovery), **file it back into the wiki** as a new page

### Lint the wiki
Periodically health-check: look for contradictions between pages, stale claims superseded by newer sources, orphan pages with no inbound links, important concepts mentioned but lacking their own page, missing cross-references, and data gaps.

## Wiki structure

```
wiki/
  index.md        — content catalog (all pages listed with links and summaries)
  log.md          — chronological record of ingests, queries, lint passes
  overview.md     — high-level project description
  architecture.md — 4-layer pipeline
  gnm-head.md     — Google GNM Head model specification
  cognitive-layer.md  — Mind (LLM orchestration)
  acoustic-layer.md   — Voice (TTS synthesis)
  temporal-alignment.md — Phonetic forced alignment
  mapping-path-a.md   — Deterministic viseme lookup (MVP)
  mapping-path-b.md   — Neural regression (FaceDiffuser-inspired)
  rendering.md        — WebGPU / Three.js rendering
  tools.md        — tool stack reference with licenses
  datasets.md     — dataset overview (VOCASET, BIWI, Multiface, MEAD)
  licensing.md    — license analysis
  roadmap.md      — phased development plan
  glossary.md     — key terms and definitions
```

## Raw sources

Immutable source documents live in `dev-docs/`. The LLM reads from them but never modifies them:
- `dev-docs/Project Dome Free Tools Research.md`
- `dev-docs/PROJECT_DOME_ROADMAP.md`

## Stack

- **LLM backend**: OpenCode (Pi) with codebase-memory-mcp for graph-based code discovery
- **Default LLM provider**: OMLX at `http://127.0.0.1:9000/v1` (OpenAI-compatible API, used for all development and wiki operations)
- **Hardware**: MacBook Pro M3 Max, 128GB unified memory
- **Wiki editor**: LLM writes markdown; user views in Obsidian or any markdown viewer
<!-- projectdome-wiki:end -->
