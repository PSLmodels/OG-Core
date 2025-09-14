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

    # Configure the root logger
    logging.basicConfig(level=level, format="%(message)s")

    # Also set the ogcore logger specifically
    logger = logging.getLogger("ogcore")
    logger.setLevel(level)

    return level


# Auto-configure when module is imported
set_logging_level(verbose=VERBOSE)
