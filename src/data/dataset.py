"""
PyTorch Dataset classes for lightning prediction.
FIXED VERSION: Removed default shape parameters to force explicit configuration.
"""

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
from ..utils.io_utils import NetCDFHandler
from .preprocessing import DataPreprocessor, TemporalProcessor

logger = logging.getLogger(__name__)

class LightningDataset(Dataset):
    """
    PyTorch Dataset for lightning prediction with multi-resolution inputs.
    """
    
    def __init__(self,
                file_pairs: List[Dict[str, Path]],
                preprocessor: DataPreprocessor,
                # MOVE REQUIRED PARAMETERS FIRST (before any optional ones)
                target_cape_shape: Tuple[int, int],        # REQUIRED - NO DEFAULT
                target_lightning_shape: Tuple[int, int],   # REQUIRED - NO DEFAULT  
                target_terrain_shape: Tuple[int, int],     # REQUIRED - NO DEFAULT
                # THEN OPTIONAL PARAMETERS (with defaults)
                sequence_length: int = 1,
                temporal_stride: int = 1,
                spatial_augmentation: Optional[callable] = None,
                meteorological_augmentation: Optional[callable] = None,
                cache_terrain: bool = True,
                # CAPE FILTERING PARAMETERS
                cape_threshold: float = 100.0,
                neighborhood_radius: int = 1,
                background_sample_ratio: float = 0.05,
                spatial_filtering: bool = True):
        """
        Initialize Lightning Dataset.
        
        Args:
            file_pairs: List of dictionaries with file paths {'cape': path, 'lightning': path, 'terrain': path}
            preprocessor: Data preprocessor instance
            sequence_length: Number of temporal steps (for future use)
            temporal_stride: Stride between temporal steps
            spatial_augmentation: Spatial augmentation function
            meteorological_augmentation: Meteorological augmentation function
            cache_terrain: Whether to cache terrain data (it's static)
            target_cape_shape: Target shape for CAPE data (lat, lon) at 25km resolution - REQUIRED
            target_lightning_shape: Target shape for lightning data (lat, lon) at 3km resolution - REQUIRED
            target_terrain_shape: Target shape for terrain data (lat, lon) at 1km resolution - REQUIRED
            cape_threshold: CAPE threshold in J/kg for filtering samples
            neighborhood_radius: Radius for spatial CAPE analysis
            background_sample_ratio: Ratio of background samples to keep
            spatial_filtering: Whether to enable spatial CAPE filtering
        """
        # VALIDATION FIX: Validate that all required shape parameters are provided
        if target_cape_shape is None or target_lightning_shape is None or target_terrain_shape is None:
            raise ValueError(
                "All target shapes are required: target_cape_shape, target_lightning_shape, target_terrain_shape. "
                "No default values are provided to force explicit configuration."
            )
        
        # VALIDATION FIX: Ensure shapes are proper tuples with 2 positive integers
        for name, shape in [("target_cape_shape", target_cape_shape), 
                           ("target_lightning_shape", target_lightning_shape), 
                           ("target_terrain_shape", target_terrain_shape)]:
            if not isinstance(shape, tuple) or len(shape) != 2:
                raise ValueError(f"{name} must be a tuple of 2 integers, got {shape}")
            if not all(isinstance(x, int) and x > 0 for x in shape):
                raise ValueError(f"{name} must contain positive integers, got {shape}")
        
        # LOGGING FIX: Log shapes for verification and debugging
        logger.info(f"Initializing LightningDataset with REQUIRED target shapes:")
        logger.info(f"  CAPE (25km): {target_cape_shape} (lat, lon)")
        logger.info(f"  Lightning (3km): {target_lightning_shape} (lat, lon)")
        logger.info(f"  Terrain (1km): {target_terrain_shape} (lat, lon)")
        
        self.file_pairs = file_pairs
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.spatial_augmentation = spatial_augmentation
        self.meteorological_augmentation = meteorological_augmentation
        self.cache_terrain = cache_terrain
        
        # SHAPE FIX: Store shapes (now guaranteed to be valid)
        self.target_cape_shape = target_cape_shape
        self.target_lightning_shape = target_lightning_shape
        self.target_terrain_shape = target_terrain_shape
        
        # CAPE FILTERING PARAMETERS
        self.cape_threshold = cape_threshold
        self.neighborhood_radius = neighborhood_radius
        self.background_sample_ratio = background_sample_ratio
        self.spatial_filtering = spatial_filtering
        
        # Cache for terrain data (static across all samples)
        self._terrain_cache = None
        
        # Build sample index with CAPE filtering
        self._build_sample_index()
        
    def _build_sample_index(self):
        """Build index of all valid samples across all files with CAPE filtering."""
        self.sample_index = []
        background_samples = []
        
        logger.info("Building sample index with CAPE filtering...")
        
        for file_idx, file_dict in enumerate(self.file_pairs):
            try:
                # Check if CAPE file exists and get time dimension
                cape_path = file_dict['cape']
                lightning_path = file_dict['lightning']
                
                # Load data to check time dimensions
                cape_ds = NetCDFHandler.load_netcdf(cape_path, variables=['cape'])
                lightning_ds = NetCDFHandler.load_netcdf(lightning_path, variables=['lightning_occurrence'])
                
                if cape_ds is None or lightning_ds is None:
                    logger.warning(f"Failed to load data for file pair {file_idx}")
                    continue
                
                # Get time dimensions
                cape_time_dim = cape_ds['cape'].shape[0] if 'time' in cape_ds.dims else 1
                lightning_time_dim = lightning_ds['lightning_occurrence'].shape[0] if 'time' in lightning_ds.dims else 1
                
                # Use minimum time dimension
                time_steps = min(cape_time_dim, lightning_time_dim)
                
                # Get CAPE values for filtering
                cape_values = cape_ds['cape'].values
                
                # Apply CAPE filtering
                for t in range(0, time_steps, self.temporal_stride):
                    time_end = min(t + self.sequence_length, time_steps)
                    
                    if time_end <= t:
                        continue
                    
                    # Get CAPE for this time step
                    if cape_values.ndim == 3:  # (time, lat, lon)
                        cape_slice = cape_values[t:time_end]
                    else:  # (lat, lon)
                        cape_slice = cape_values
                    
                    # Calculate mean CAPE for this sample
                    mean_cape = np.nanmean(cape_slice)
                    
                    sample = {
                        'file_idx': file_idx,
                        'time_start': t,
                        'time_end': time_end,
                        'mean_cape': mean_cape
                    }
                    
                    # CAPE-based filtering (use configurable threshold)
                    if mean_cape >= self.cape_threshold:  # Use instance variable instead of hardcoded 100.0
                        self.sample_index.append(sample)
                    else:
                        background_samples.append(sample)
                
                cape_ds.close()
                lightning_ds.close()
                
            except Exception as e:
                logger.warning(f"Error processing file pair {file_idx}: {e}")
                continue
        
        # Add configurable percentage of background samples randomly
        background_keep = int(len(background_samples) * self.background_sample_ratio)  # Use instance variable instead of hardcoded 0.05
        if background_keep > 0:
            random.shuffle(background_samples)
            self.sample_index.extend(background_samples[:background_keep])
        
        logger.info(f"CAPE filtering results:")
        logger.info(f"  High CAPE samples (≥{self.cape_threshold} J/kg): {len(self.sample_index) - background_keep}")
        logger.info(f"  Background samples kept: {background_keep}")
        logger.info(f"  Total samples: {len(self.sample_index)}")
        logger.info(f"  Filtering removed: {len(background_samples) - background_keep:,} low-CAPE samples")
        
        if len(self.sample_index) == 0:
            raise RuntimeError("No valid samples found in dataset after CAPE filtering")
    
    def _load_terrain(self, terrain_path: Path) -> torch.Tensor:
        """Load and cache terrain data at native 1km resolution."""
        if self.cache_terrain and self._terrain_cache is not None:
            return self._terrain_cache
        
        try:
            terrain_ds = NetCDFHandler.load_netcdf(terrain_path, variables=['elevation'])
            
            if terrain_ds is None:
                raise RuntimeError(f"Failed to load terrain data from {terrain_path}")
            
            terrain_data = terrain_ds['elevation']
            terrain_tensor = self.preprocessor.preprocess_terrain(terrain_data, self.target_terrain_shape)
            
            terrain_ds.close()
            
            if self.cache_terrain:
                self._terrain_cache = terrain_tensor
            
            return terrain_tensor
            
        except Exception as e:
            logger.error(f"Error loading terrain data: {e}")
            # Return zeros as fallback
            return torch.zeros(1, *self.target_terrain_shape)
    
    def _load_cape(self, cape_path: Path, time_start: int, time_end: int) -> torch.Tensor:
        """Load CAPE data at native 25km resolution."""
        try:
            cape_ds = NetCDFHandler.load_netcdf(cape_path, variables=['cape'])
            
            if cape_ds is None:
                raise RuntimeError(f"Failed to load CAPE data from {cape_path}")
            
            # Select time slice
            if 'time' in cape_ds.dims:
                cape_data = cape_ds['cape'].isel(time=slice(time_start, time_end))
            else:
                cape_data = cape_ds['cape']
                cape_data = cape_data.expand_dims('time')
            
            cape_tensor = self.preprocessor.preprocess_cape(cape_data)
            
            cape_ds.close()
            
            return cape_tensor
            
        except Exception as e:
            logger.error(f"Error loading CAPE data: {e}")
            # Return zeros as fallback
            return torch.zeros(time_end - time_start, 1, *self.target_cape_shape)
    
    def _load_lightning(self, lightning_path: Path, time_start: int, time_end: int) -> torch.Tensor:
        """Load lightning data for specified time range."""
        try:
            lightning_ds = NetCDFHandler.load_netcdf(lightning_path, variables=['lightning_occurrence'])
            
            if lightning_ds is None:
                raise RuntimeError(f"Failed to load lightning data from {lightning_path}")
            
            # Select time slice
            if 'time' in lightning_ds.dims:
                lightning_data = lightning_ds['lightning_occurrence'].isel(time=slice(time_start, time_end))
            else:
                lightning_data = lightning_ds['lightning_occurrence']
                lightning_data = lightning_data.expand_dims('time')
            
            lightning_tensor = self.preprocessor.preprocess_lightning(lightning_data)
            
            lightning_ds.close()
            
            return lightning_tensor
            
        except Exception as e:
            logger.error(f"Error loading lightning data: {e}")
            # Return zeros as fallback
            return torch.zeros(time_end - time_start, *self.target_lightning_shape)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing input and target tensors at native resolutions
        """
        sample_info = self.sample_index[idx]
        file_dict = self.file_pairs[sample_info['file_idx']]
        
        # Load data at native resolutions
        cape_tensor = self._load_cape(file_dict['cape'], 
                                     sample_info['time_start'], 
                                     sample_info['time_end'])
        
        lightning_tensor = self._load_lightning(file_dict['lightning'],
                                              sample_info['time_start'],
                                              sample_info['time_end'])
        
        terrain_tensor = self._load_terrain(file_dict['terrain'])
        
        # For single timestep, remove time dimension
        if self.sequence_length == 1:
            cape_tensor = cape_tensor.squeeze(0)  # (1, lat, lon)
            lightning_tensor = lightning_tensor.squeeze(0)  # (lat, lon)
        
        # Apply augmentations
        if self.spatial_augmentation is not None:
            cape_tensor, lightning_tensor, terrain_tensor = self.spatial_augmentation(
                cape_tensor, lightning_tensor, terrain_tensor
            )
        
        if self.meteorological_augmentation is not None:
            cape_tensor = self.meteorological_augmentation(cape_tensor)
        
        sample = {
            'cape': cape_tensor,
            'terrain': terrain_tensor,
            'lightning': lightning_tensor,
            'file_idx': sample_info['file_idx'],
            'time_start': sample_info['time_start']
        }
        
        return sample

class ERA5Dataset(LightningDataset):
    """
    Extended dataset class for future ERA5 3D data.
    Currently inherits from LightningDataset, will be expanded later.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize ERA5 dataset (placeholder for future implementation)."""
        super().__init__(*args, **kwargs)
        # Future: Add ERA5-specific processing
        
    def _load_era5(self, era5_path: Path, time_start: int, time_end: int) -> torch.Tensor:
        """Load ERA5 3D data (placeholder for future implementation)."""
        # Future implementation for 3D ERA5 data
        # Will handle multiple pressure levels and variables
        pass