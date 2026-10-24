# Groundwork

Groundwork improves an existing LLM on three common problems:

1. **It knows things but can't get the task done.** It has the facts, but doesn't plan, compute, check its work or use files.
2. **It hallucinates.** It answers confidently when it doesn't actually know.
3. **Its tone is inconsistent.** Replies drift from the voice you want.

It works with any chat model: your own, an open-weight model, or anything served behind an OpenAI-compatible API. It handles text, images, video frames and documents.

There are two halves. You can use either one alone:

| | Runtime layer (no training) | Fine-tuning kit |
|---|---|---|
| What it is | A wrapper around the model at inference time | Builds verified training data and fine-tunes the model |
| Task know-how | Tool-using agent (Python, search, file reader) that works step by step and votes between candidate actions | Trains on the model's own **verified** successful runs (rejection sampling); recovered errors are kept so it learns to fix mistakes |
| Hallucination | Consistency guard: samples several answers. When they disagree, it verifies with tools or says it isn't sure | Probes what the model knows. It reinforces known facts, teaches lookup or "I'm not sure" for unknown ones, and never trains on facts the model doesn't know |
| Tone | Style-guide prompt, rule checks and an automatic rewrite pass | Your hand-written examples plus the model's own correct answers rewritten into your voice |
| Gains | Immediate, cost per request | Permanent, no extra inference cost |

Both halves are measured by the same evaluation: task accuracy, hallucination rate, abstention rate, selective accuracy and tone score. You can compare any run against a baseline.

---

## Install

```bash
pip install -e .                 # runtime layer + data tools
pip install -e ".[train,lora]"   # + fine-tuning
pip install -e ".[all]"          # + multi-GPU, video and PDF/DOCX readers
```

Serve your model behind an OpenAI-compatible endpoint, for example with vLLM:

```bash
vllm serve /path/to/your-model --served-model-name my-model --port 8000
```

Copy `configs/example.yaml` and edit the `model:` section.

```bash
groundwork check -c configs/my.yaml --ping     # config, data-source terms, connectivity
```

## Use the runtime layer

```bash
groundwork chat  -c configs/my.yaml            # interactive
groundwork serve -c configs/my.yaml            # OpenAI-compatible server on :8080
```

Point any app that speaks the OpenAI chat API at `http://localhost:8080/v1`. Each reply carries a `groundwork` field showing the route taken (`direct`, `agent`, `verified` or `abstain`).

How a request flows:

```
request ─> route ─┬─> direct answer ─> consistency guard ─┬─> agree: keep
                  │                                       ├─> disagree: verify with tools
                  │                                       └─> still unsure: say so
                  └─> agent (plan → code → observe ...) ─> compose reply
                            every reply ─> tone check ─> rewrite if it breaks the style guide
```

## Use the fine-tuning kit

```bash
groundwork eval     -c configs/my.yaml --label baseline   # 0. measure first
groundwork probe    -c configs/my.yaml                    # 1. what does the model know?
groundwork generate -c configs/my.yaml                    # 2. agent runs, scored against gold answers
groundwork build    -c configs/my.yaml                    # 3. filter, shape and mix -> work/sft.jsonl
groundwork train    -c configs/my.yaml                    # 4. fine-tune (LoRA or full)
# serve the fine-tuned weights, then:
groundwork eval     -c configs/my.yaml --label after-sft --set eval.baseline_report=reports/baseline/report.json
```

`generate` resumes where it stopped. `eval` exits non-zero if any metric regresses beyond `eval.regression_tolerance`, so it can gate CI.

What `build` puts into the mix (`work/sft_stats.json` shows the counts):

| Source | From | Purpose |
|---|---|---|
| `trace` | Agent runs that reached the **correct** answer, with no parse errors and no step-limit hits; shortest run per task | Task know-how |
| `closed_book_known` / `_partial` | The model's own correct answers to questions it mostly gets right | Reinforce real knowledge |
| `abstain` | "I'm not sure" replies for questions it can't answer, capped at `max_abstain_fraction` (default 15%) to avoid over-refusal | Fewer hallucinations |
| `tone` | Your hand-written examples (any that break the style guide are skipped) | Voice |
| `replay` | A sample of your original training data | Keep existing skills |

Eval questions listed in `data.decontaminate_against` are removed from the mix (exact and 13-gram overlap).

Multi-GPU: `accelerate launch -m groundwork.cli train -c configs/my.yaml`.
Vision-language models: set `train.multimodal: true`. Vision towers and projectors are frozen by default (`train.freeze_modules`).
Custom architectures: set `train.loader: "your_pkg.module:load"`, a function returning `(model, tokenizer, processor)`.

## Data formats

**Tasks** (`data.tasks`, `eval.tasks`), one JSON object per line:

```json
{"id": "math-1", "question": "A shop sells 3 notebooks for $7.50. How much do 17 cost?", "answer": "42.5", "answer_type": "numeric", "domain": "math"}
{"id": "fact-1", "question": "What is the highest mountain in Africa?", "answer": ["Kilimanjaro"], "answer_type": "contains", "closed_book": true}
{"id": "vis-1",  "question": "Which quarter has the tallest bar?", "answer": "Q2", "images": ["charts/q.png"]}
{"id": "doc-1",  "question": "Total stock value?", "answer": "1120.53", "answer_type": "numeric", "files": ["inventory.csv"]}
{"id": "code-1", "question": "Write is_palindrome(s).", "answer_type": "code", "tests": "assert is_palindrome('abba')"}
```

- `answer_type` is one of `exact`, `contains`, `numeric`, `choice`, `code` (runs `tests`), or `judge` (uses the `judge:` endpoint).
- `closed_book: true` marks questions that can be answered from memory. These are the ones the knowledge probe uses.
- `videos` are sampled into frames (`data.video_frames`).

**Style guide** (`runtime.tone.style_guide`): see [`examples/style_guide.yaml`](examples/style_guide.yaml). It sets voice, do/don't lists, banned phrases, a length limit, an emoji rule, and the wording used when the model isn't sure.

**Tone examples** (`data.tone_examples`): `{"prompt": ..., "response": ...}` per line, written by people.

**Messages** everywhere use an OpenAI-like shape. Content can be a string, or a list of parts: `{"type": "text"|"image"|"video"|"file", ...}`.

## Where training data may come from

Groundwork checks every model whose outputs become training data. That covers trace generation, probing, restyling and judging.

- **Your own model** (`ownership: self`) is the default. The model improves from its own verified work, and no outside model is needed.
- **A third-party model** (`ownership: third_party`) can be used only when its license or terms allow training other models on its outputs. You must set `provider`, `terms_url` and `distillation_permitted: true`. Open-weight models with permissive licenses are a good fit.
- **Refused:** Anthropic, OpenAI and Google Gemini APIs. Their terms prohibit using them to develop competing models. See [`groundwork/compliance.py`](groundwork/compliance.py).

Every data source used is logged to `work/provenance.jsonl`. The runtime layer and evaluation can wrap any model.

## Safety note on code execution

Agent code runs in a separate Python process with a time limit and a memory limit. Tool calls go back through the parent process, and `read_file` is confined to `agent.files_root`. **This is not a security sandbox.** Run Groundwork inside a container when tasks or models are untrusted.

## Development

```bash
pip install -e ".[train,dev]"
pytest -q
```

The tests run on CPU with scripted models and a tiny randomly initialised transformer. No downloads or GPUs are needed.

## Background and acknowledgements

Groundwork is an independent implementation. Its design draws on:

- *Distilling LLM Agents into Small Models with Retrieval and Code Tools* (Kang et al., 2025, arXiv:2505.17612) and its open-source code: agent trajectories with code and retrieval tools, correctness filtering, first-thought prefix, and self-consistent action generation.
- [DistillKit](https://github.com/arcee-ai/distillkit) and [LLM-Distillery](https://github.com/golololologol/LLM-Distillery): practical distillation pipelines and lessons on loss masking, resumable data collection and evaluation.
- Gekhman et al., *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* (2024), and Zhang et al., *R-Tuning* (2024): why unknown facts should become abstention or lookup instead of answers.
- Manakul et al., *SelfCheckGPT* (2023), and Wang et al., *Self-Consistency* (2023): agreement between samples as a hallucination signal.
- Zelikman et al., *STaR* (2022): self-improvement from verified outputs.
- Wang et al., *CodeAct* (2024), and Hugging Face smolagents: code as the action format.

## License

MIT. See [LICENSE](LICENSE).
