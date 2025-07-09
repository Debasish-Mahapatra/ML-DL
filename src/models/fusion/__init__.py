"""
Fusion modules for combining multi-modal and multi-scale data.
"""

from .meteorological_fusion import MeteorologicalFusion
from .multiscale_fusion import MultiScaleFusion, TerrainGuidedUpsampling

__all__ = [
    "MeteorologicalFusion",
    "MultiScaleFusion", 
    "TerrainGuidedUpsampling"
]