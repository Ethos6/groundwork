"""Model connectors and a factory that applies the data-source terms check."""

from __future__ import annotations

from ..compliance import check_data_source, record_provenance
from ..config import Config, EndpointConfig
from .base import ChatModel, GenerationError, trim_at_stop  # noqa: F401


def build_model(ep: EndpointConfig, cfg: Config) -> ChatModel:
    if ep.kind == "openai":
        from .openai_compat import OpenAICompatModel

        return OpenAICompatModel(ep, media_root=cfg.data.media_root, video_frames=cfg.data.video_frames)
    if ep.kind == "hf":
        from .hf import HFChatModel

        return HFChatModel(
            ep.path or ep.model, dtype=ep.dtype, device=ep.device, trust_remote_code=ep.trust_remote_code,
            multimodal=ep.multimodal, media_root=cfg.data.media_root, video_frames=cfg.data.video_frames,
            max_tokens=ep.max_tokens,
        )
    raise ValueError(f"Unknown endpoint kind {ep.kind!r} (expected 'openai' or 'hf')")


def endpoint_for(cfg: Config, role: str) -> EndpointConfig:
    ep = getattr(cfg, role)
    if ep is None:
        raise ValueError(f"No '{role}' endpoint configured")
    return ep


def build_data_source(cfg: Config, role: str, purpose: str) -> ChatModel:
    """Build a model whose outputs will become training data (terms-checked and recorded)."""
    ep = endpoint_for(cfg, role)
    check_data_source(ep, role)
    record_provenance(cfg.data.work_dir, ep, role, purpose)
    return build_model(ep, cfg)


def build_judge(cfg: Config) -> ChatModel | None:
    if cfg.judge is None:
        return None
    check_data_source(cfg.judge, "judge")
    return build_model(cfg.judge, cfg)
