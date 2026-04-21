"""
logger_setup.py
===============
Shared logger for the vision pipeline.  All v3 modules import from here.
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("vision_pipeline")
