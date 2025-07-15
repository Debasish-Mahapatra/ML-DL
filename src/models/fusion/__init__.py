"""
Fusion modules for combining multi-modal and multi-scale data.

Updated for Strategy 3: Multi-Resolution Learning approach.
This replaces the old multiscale_fusion.py with the new multi-resolution approach.
"""

# Import the new multi-resolution fusion modules
from .multi_resolution_fusion import MultiResolutionFusion
from .multi_resolution_terrain_processor import MultiResolutionTerrainProcessor
from .multi_resolution_meteorological_processor import MultiResolutionMeteorologicalProcessor

# Keep the existing meteorological fusion (still useful for CAPE+ERA5)
from .meteorological_fusion import MeteorologicalFusion

# Remove the old problematic imports:
# from .multiscale_fusion import MultiScaleFusion, TerrainGuidedUpsampling  # DELETE THESE

__all__ = [
    # New multi-resolution approach
    "MultiResolutionFusion",
    "MultiResolutionTerrainProcessor", 
    "MultiResolutionMeteorologicalProcessor",
    
    # Existing meteorological fusion (kept for CAPE+ERA5 combination)
    "MeteorologicalFusion",
    
    # Removed old problematic classes:
    # "MultiScaleFusion",     # DELETE - caused shape mismatch
    # "TerrainGuidedUpsampling"  # DELETE - problematic upsampling
]