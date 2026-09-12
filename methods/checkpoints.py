"""Small trusted-local PyTorch checkpoint helpers, shared by the trainers.

Training state belongs to each method; this module only handles files and RNG.
Checkpoint files can contain Python objects. Load only artifacts you trust.
"""
from dataclasses import asdict
from pathlib import Path

import torch
from hedging_gym.config import config_from_dict, market_from_dict


def saved_market(value):
    """Recover pre-scheme pickled objects without inheriting today's QE-M default."""
    if value is None:
        return None
    # asdict/getattr would materialize a newly added class default on old objects.
    fields = value if isinstance(value, dict) else dict(vars(value), model=value.model)
    return market_from_dict(fields)


def saved_config(value):
    """Normalize trusted-local dictionary or historical dataclass contracts."""
    if isinstance(value, dict):
        return config_from_dict(value)
    # A field added with default_factory (settlement) has no class fallback.
    # Only materialize top-level fields present in the old instance.
    fields = {name: asdict(item) for name, item in vars(value).items()}
    fields["market"] = asdict(saved_market(value.market))
    return config_from_dict(fields)


def rng_state():
    return dict(cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, payload):
    """Replace the latest snapshot atomically; retain the first completed step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    if payload["step"] == 1:
        early = path.with_name(path.stem + "-early" + path.suffix)
        if not early.exists():
            torch.save(payload, early)


def load_checkpoint(path, *, method, config):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # New optional config fields must not invalidate an unchanged older task.
    if saved["method"] != method or saved_config(saved["config"]) != config:
        raise ValueError("checkpoint method or financial configuration differs")
    return saved


def check_resume_options(saved, options):
    """A run may extend its total updates without silently changing its recipe."""
    if options["updates"] < saved["step"]:
        raise ValueError("requested updates precede the saved completed step")
    previous = {key: value for key, value in saved["options"].items() if key != "updates"}
    current = {key: value for key, value in options.items() if key != "updates"}
    if previous != current:
        raise ValueError("resume requires the saved training options except total updates")


def due_checkpoint(step, total, every):
    return step == 1 or step == total or step % every == 0
