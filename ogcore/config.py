"""
Centralized configuration for OG-Core logging.

This module provides a centralized way to configure logging levels
across all OG-Core modules, avoiding circular import issues.
"""

import logging

VERBOSE = True  # Default verbosity setting


def set_logging_level(verbose=True):
    """
    Set the logging level for OG-Core modules.

    Args:
        verbose (bool): If True, set logging to INFO level.
                       If False, set logging to WARNING level.

    Returns:
        int: The logging level that was set.
    """
    global VERBOSE
    VERBOSE = verbose
    level = logging.INFO if verbose else logging.WARNING

    # Configure the ogcore logger with an explicit StreamHandler so it
    # works regardless of whether another package (e.g. distributed/dask)
    # has already called basicConfig() on the root logger.
    logger = logging.getLogger("ogcore")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False

    return level


# Auto-configure when module is imported
set_logging_level(verbose=VERBOSE)
