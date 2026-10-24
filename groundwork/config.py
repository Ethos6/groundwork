"""YAML configuration loaded into typed dataclasses.

Unknown keys raise an error so typos are caught early. Values can be overridden from the
command line with ``--set section.key=value`` (the value is parsed as YAML).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------- endpoints
@dataclass
class EndpointConfig:
    """A chat model reachable over an OpenAI-compatible API (``kind: openai``) or loaded
    locally with Hugging Face Transformers (``kind: hf``)."""

    kind: str = "openai"  # openai | hf
    model: str = ""  # model name sent to the API (kind=openai)
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = ""  # name of the env var holding the API key (optional)
    # Who owns the model. "self" = your own model. "third_party" = someone else's model;
    # it can only be used to create training data if its terms allow that (see compliance.py).
    ownership: str = "self"  # self | third_party
    provider: str = ""
    terms_url: str = ""
    distillation_permitted: bool = False
    supports_prefill: bool = False  # vLLM-style continue_final_message
    supports_n: bool = True  # server accepts n>1 in one request
    max_tokens: int = 1024
    timeout_s: float = 120.0
    max_retries: int = 4
    extra_body: dict = field(default_factory=dict)
    # kind=hf
    path: str = ""
    dtype: str = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False
    multimodal: bool = False


# --------------------------------------------------------------------------- agent
@dataclass
class AgentConfig:
    max_steps: int = 6
    tools: list = field(default_factory=lambda: ["python", "search", "read_file"])
    corpus: str = ""  # local JSONL corpus ({"id","title","text"}) for the search tool
    search_url: str = ""  # optional HTTP retriever: POST {"query","top_k"} -> {"results":[...]}
    search_top_k: int = 3
    files_root: str = "."  # read_file() is restricted to this directory
    python_timeout_s: float = 10.0
    python_memory_mb: int = 1024
    max_observation_chars: int = 3000
    temperature: float = 0.0
    self_consistency_n: int = 1  # >1 enables self-consistent action selection
    self_consistency_temperature: float = 0.4
    force_final_on_max_steps: bool = True


# --------------------------------------------------------------------------- runtime
@dataclass
class GuardConfig:
    enabled: bool = True
    fact_check_gate: bool = True  # ask the model whether the request depends on facts
    samples: int = 5
    temperature: float = 0.8
    accept_agreement: float = 0.6  # >= this share of samples agreeing -> answer directly
    verify_with_agent: bool = True  # low agreement -> try to verify with tools first


@dataclass
class ToneConfig:
    style_guide: str = ""  # path to a style guide YAML (see examples/style_guide.yaml)
    rewrite: bool = True
    max_rewrites: int = 1


@dataclass
class RuntimeConfig:
    agent_mode: str = "auto"  # auto | always | never
    compose_reply: bool = True  # turn the agent's final_answer into a full reply
    guard: GuardConfig = field(default_factory=GuardConfig)
    tone: ToneConfig = field(default_factory=ToneConfig)
    host: str = "127.0.0.1"
    port: int = 8080


# --------------------------------------------------------------------------- data
@dataclass
class DataConfig:
    tasks: str = "examples/tasks.jsonl"
    work_dir: str = "work"
    media_root: str = "."
    video_frames: int = 8
    workers: int = 4
    # trace generation (rejection sampling)
    actor: str = "model"  # model | teacher
    attempts_per_task: int = 4
    attempt_temperature: float = 0.7
    keep_per_task: int = 1
    keep_recovered_errors: bool = True
    first_thought_prefix: bool = False
    # knowledge probe / hallucination shaping
    probe_samples: int = 5
    probe_temperature: float = 0.8
    known_threshold: float = 0.6
    unknown_threshold: float = 0.2
    abstain_when_unknown: bool = True
    max_abstain_fraction: float = 0.15
    # tone
    tone_examples: str = ""  # JSONL of {"prompt","response"} written by your team
    restyle_answers: bool = True
    # share of closed-book/tone examples trained WITHOUT the style-guide system prompt,
    # so the voice becomes the model's default rather than depending on the prompt
    system_prompt_dropout: float = 0.5
    # mixing
    replay: str = ""  # JSONL of {"messages":[...]} from the model's original training mix
    replay_fraction: float = 0.2
    decontaminate_against: list = field(default_factory=list)
    seed: int = 42


# --------------------------------------------------------------------------- training
@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class TrainConfig:
    base_model: str = ""  # local path or hub id of the model being improved
    loader: str = ""  # optional "package.module:function" returning (model, tokenizer, processor)
    data: str = ""  # defaults to <work_dir>/sft.jsonl
    output_dir: str = "runs/sft"
    max_length: int = 4096
    lr: float = 1.0e-5
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"  # cosine | linear | constant
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    multimodal: bool = False
    freeze_modules: list = field(
        default_factory=lambda: ["vision_tower", "visual", "multi_modal_projector", "vision_model", "image_encoder"]
    )
    trust_remote_code: bool = False
    tool_role_as_user: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    save_every_steps: int = 0
    log_every_steps: int = 10
    seed: int = 42


# --------------------------------------------------------------------------- eval
@dataclass
class EvalConfig:
    tasks: list = field(default_factory=list)
    modes: list = field(default_factory=lambda: ["direct", "assistant"])
    output_dir: str = "reports"
    baseline_report: str = ""
    regression_tolerance: float = 0.02
    workers: int = 4


# --------------------------------------------------------------------------- root
@dataclass
class Config:
    model: EndpointConfig = field(default_factory=EndpointConfig)
    teacher: Optional[EndpointConfig] = None
    judge: Optional[EndpointConfig] = None
    agent: AgentConfig = field(default_factory=AgentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- loading
def _unwrap_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _coerce(tp: Any, value: Any, where: str) -> Any:
    """YAML reads '2e-5' as a string and users write '8' for floats; fix scalar types."""
    if value is None:
        return value
    try:
        if tp is float and isinstance(value, (int, str)) and not isinstance(value, bool):
            return float(value)
        if tp is int and isinstance(value, str):
            return int(value)
        if tp is bool and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes", "1", "on"):
                return True
            if low in ("false", "no", "0", "off"):
                return False
            raise ValueError(value)
    except ValueError as e:
        raise ValueError(f"{where}: cannot read {value!r} as {tp.__name__}") from e
    return value


def _build(cls: type, data: Any, where: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        tp = _unwrap_optional(hints[name])
        if dataclasses.is_dataclass(tp) and value is not None:
            kwargs[name] = _build(tp, value, f"{where}.{name}")
        else:
            kwargs[name] = _coerce(tp, value, f"{where}.{name}")
    return cls(**kwargs)


def _set_path(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        if node.get(k) is None:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        _set_path(raw, key.strip(), yaml.safe_load(val))
    return _build(Config, raw, "config")
