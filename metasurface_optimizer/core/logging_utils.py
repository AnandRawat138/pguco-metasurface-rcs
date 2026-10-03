"""Logging configuration."""
from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; configures a sane default handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
