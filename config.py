"""Load settings from config.toml, the only place settings live."""
import logging
import tomllib
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path=None) -> dict:
    path = Path(path) if path else ROOT / "config.toml"
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    # Resolve relative paths against the project folder and create their folders.
    for key, value in cfg["paths"].items():
        p = Path(value)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(p)
    return cfg


def setup_logging(cfg: dict) -> None:
    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return  # already set up
    handler = RotatingFileHandler(
        cfg["paths"]["log_file"], maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
