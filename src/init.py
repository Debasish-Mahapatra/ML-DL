"""
Lightning Prediction System

A hybrid deep learning framework for lightning prediction using multi-scale fusion
of meteorological and terrain data with physics-informed constraints.
"""

__version__ = "0.1.0"
__author__ = "Lightning Prediction Team"

from .models import LightningPredictor
from .data import LightningDataset, LightningDataModule
from .training import LightningTrainer

__all__ = [
    "LightningPredictor",
    "LightningDataset", 
    "LightningDataModule",
    "LightningTrainer"
]
