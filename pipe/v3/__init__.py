# v3 — modular AI reasoning pipeline package
# Each sub-module has a single responsibility:
#
#   config.py        — constants and shared settings
#   data_loader.py   — load JSON logs from disk
#   image_sampler.py — sample + semantically deduplicate images
#   embeddings.py    — pixel-level embedding extraction + cosine search
#   rag.py           — RAG store: index past failures, retrieve similar cases
#   schemas.py       — Pydantic models for structured LLM output
#   tools.py         — agent tool functions (stats queries)
#   chains.py        — LLM construction + prompt builders
#   graph.py         — LangGraph-style agentic state machine
#   renderer.py      — Markdown report rendering + dynamic recommendations
#   server_utils.py  — vLLM server health polling
