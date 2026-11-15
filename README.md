# AI Engineering

Learning AI engineering from LLM internals to serving to agents — notes and implementations.

Notes are authored as [Excalidraw](https://excalidraw.com) diagrams; implementations live in Jupyter notebooks per module.

## Curriculum

**Part I — LLM Internals and Training (Modules 0–6)**
- Module 0: Sequence modelling and Transformers — pre/post-norm, RMSNorm, SwiGLU/GeGLU, parameter and FLOP counting, KV cache math, causal mask, teacher forcing
- Module 1: Tokenization and data
- Module 2: Positional encoding and long context
- Module 3: Modern architecture internals
- Module 4: Pretraining at scale
- Module 5: Post-training and alignment
- Module 6: Quantization and fine-tuning

**Part II — Inference Engineering and Serving (Module 7)**
- Inference fundamentals, engines (vLLM, SGLang, TensorRT-LLM), distributed and production serving

**Part III — Reasoning and Agentic AI Engineering (Modules 8–12)**
- Reasoning models, retrieval and context engineering, agents

## Repository layout

```
notes/          Excalidraw diagrams, one folder per module
notebooks/      Hands-on notebooks per module
projects/       Per-module projects (e.g. nanoGPT with RMSNorm + SwiGLU)
```

## Running notebooks

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab
```
