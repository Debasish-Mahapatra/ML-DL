"""
Data preprocessing utilities for meteorological and lightning data.
"""

import numpy as np
import xarray as xr
import torch
from typing import Dict, Tuple, Optional, List
from scipy import ndimage
from skimage.transform import resize
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles preprocessing of meteorological and terrain data."""
    
    def __init__(self, 
                 cape_stats: Optional[Dict[str, float]] = None,
                 terrain_stats: Optional[Dict[str, float]] = None):
        """
        Initialize preprocessor with normalization statistics.
        
        Args:
            cape_stats: Dictionary with 'mean' and 'std' for CAPE normalization
            terrain_stats: Dictionary with 'mean' and 'std' for terrain normalization
        """
        self.cape_stats = cape_stats or {'mean': 0.0, 'std': 1.0}
        self.terrain_stats = terrain_stats or {'mean': 0.0, 'std': 1.0}
        
    def preprocess_cape(self, cape_data: xr.DataArray) -> torch.Tensor:
        """
        Preprocess CAPE data.
        
        Args:
            cape_data: CAPE data from NetCDF (time, lat, lon)
            
        Returns:
            Preprocessed CAPE tensor (time, 1, lat, lon)
        """
        # Convert to numpy
        cape_np = cape_data.values
        
        # Handle missing values
        cape_np = np.nan_to_num(cape_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values (CAPE should be >= 0)
        cape_np = np.clip(cape_np, 0, 10000)  # Max reasonable CAPE ~10,000 J/kg
        
        # Normalize
        cape_normalized = (cape_np - self.cape_stats['mean']) / self.cape_stats['std']
        
        # Convert to tensor and add channel dimension
        cape_tensor = torch.from_numpy(cape_normalized).float()
        
        # Add channel dimension: (time, lat, lon) -> (time, 1, lat, lon)
        cape_tensor = cape_tensor.unsqueeze(1)
        
        return cape_tensor
    
    def preprocess_terrain(self, terrain_data: xr.DataArray, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess terrain data and handle resolution conversion.
        
        Args:
            terrain_data: Terrain elevation data (lat, lon)
            target_shape: Target spatial shape (height, width)
            
        Returns:
            Preprocessed terrain tensor (1, height, width)
        """
        # Convert to numpy
        terrain_np = terrain_data.values
        
        # Handle missing values
        terrain_np = np.nan_to_num(terrain_np, nan=0.0)
        
        # Resize to target shape if needed
        if terrain_np.shape != target_shape:
            terrain_np = resize(terrain_np, target_shape, preserve_range=True, anti_aliasing=True)
        
        # Normalize
        terrain_normalized = (terrain_np - self.terrain_stats['mean']) / self.terrain_stats['std']
        
        # Convert to tensor and add channel dimension
        terrain_tensor = torch.from_numpy(terrain_normalized).float()
        terrain_tensor = terrain_tensor.unsqueeze(0)  # Add channel dimension
        
        return terrain_tensor
    
    def preprocess_lightning(self, lightning_data: xr.DataArray) -> torch.Tensor:
        """
        Preprocess lightning ground truth data.
    
        Args:
            lightning_data: Lightning occurrence data (time, lat, lon)
        
        Returns:
            Lightning tensor (time, lat, lon)
        """
        # Convert to numpy
        lightning_np = lightning_data.values
    
        # Check and fix dimension order if needed
        # Expected: (time, longitude, latitude) -> shape should be (time, 221, 181)
        if lightning_np.ndim == 3 and lightning_np.shape[-2:] == (181, 221):
            # Data is loaded as (time, latitude, longitude) - need to transpose
            lightning_np = lightning_np.transpose(0, 2, 1)  # -> (time, longitude, latitude)
            print(f"Fixed lightning dimension order: {lightning_data.shape} -> {lightning_np.shape}")
    
        # Handle missing values (assume no lightning for missing data)
        lightning_np = np.nan_to_num(lightning_np, nan=0.0)
    
        # Ensure binary values (0 or 1)
        lightning_np = np.clip(lightning_np, 0, 1)
    
        # Convert to tensor
        lightning_tensor = torch.from_numpy(lightning_np).float()
    
        return lightning_tensor
    
    def compute_normalization_stats(self, file_paths: List[str], data_type: str) -> Dict[str, float]:
        """
        Compute normalization statistics from multiple files.
        
        Args:
            file_paths: List of NetCDF file paths
            data_type: Type of data ('cape' or 'terrain')
            
        Returns:
            Dictionary with mean and std
        """
        logger.info(f"Computing normalization statistics for {data_type}")
        
        all_values = []
        
        for file_path in file_paths:
            try:
                ds = xr.open_dataset(file_path)
                
                if data_type == 'cape':
                    var_name = 'cape'  # Update with actual variable name
                elif data_type == 'terrain':
                    var_name = 'elevation'  # Update with actual variable name
                else:
                    raise ValueError(f"Unknown data type: {data_type}")
                
                if var_name in ds.data_vars:
                    values = ds[var_name].values
                    values = np.nan_to_num(values, nan=0.0)
                    
                    if data_type == 'cape':
                        values = np.clip(values, 0, 10000)
                    
                    all_values.append(values.flatten())
                
                ds.close()
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        if not all_values:
            logger.warning(f"No valid data found for {data_type}, using default stats")
            return {'mean': 0.0, 'std': 1.0}
        
        # Concatenate all values
        all_values = np.concatenate(all_values)
        
        # Compute statistics
        mean = float(np.mean(all_values))
        std = float(np.std(all_values))
        
        # Ensure std is not zero
        if std < 1e-8:
            std = 1.0
            logger.warning(f"Standard deviation too small for {data_type}, using 1.0")
        
        stats = {'mean': mean, 'std': std}
        logger.info(f"{data_type} normalization stats: {stats}")
        
        return stats

class ResolutionConverter:
    """Handles conversion between different spatial resolutions."""
    
    @staticmethod
    def upsample_nearest(data: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Upsample data using nearest neighbor interpolation.
        
        Args:
            data: Input data array (..., height, width)
            scale_factor: Upsampling scale factor
            
        Returns:
            Upsampled data
        """
        if len(data.shape) < 2:
            raise ValueError("Data must have at least 2 spatial dimensions")
        
        # Get spatial dimensions (last 2)
        *other_dims, height, width = data.shape
        
        # Calculate new size
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Upsample
        upsampled = ndimage.zoom(data, 
                               [1] * len(other_dims) + [scale_factor, scale_factor], 
                               order=0)  # nearest neighbor
        
        return upsampled
    
    @staticmethod  
    def downsample_average(data: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Downsample data using block averaging.
        
        Args:
            data: Input data array (..., height, width)
            scale_factor: Downsampling scale factor (< 1)
            
        Returns:
            Downsampled data
        """
        if scale_factor >= 1:
            raise ValueError("Scale factor must be < 1 for downsampling")
        
        if len(data.shape) < 2:
            raise ValueError("Data must have at least 2 spatial dimensions")
        
        # Get spatial dimensions (last 2)
        *other_dims, height, width = data.shape
        
        # Calculate new size
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Use scipy zoom with linear interpolation for averaging effect
        downsampled = ndimage.zoom(data,
                                 [1] * len(other_dims) + [scale_factor, scale_factor],
                                 order=1)  # linear interpolation
        
        return downsampled
    
    @staticmethod
    def align_grids(data_25km: np.ndarray, 
                   data_3km: np.ndarray,
                   data_1km: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align different resolution grids to common reference.
        
        Args:
            data_25km: 25km resolution data
            data_3km: 3km resolution data  
            data_1km: 1km resolution data
            
        Returns:
            Tuple of aligned data arrays
        """
        # For now, we'll handle this in the model architecture
        # This function can be extended for more sophisticated grid alignment
        return data_25km, data_3km, data_1km

class TemporalProcessor:
    """Handles temporal aspects of the data."""
    
    @staticmethod
    def extract_temporal_features(time_coord: xr.DataArray) -> Dict[str, np.ndarray]:
        """
        Extract temporal features from time coordinate.
        
        Args:
            time_coord: Time coordinate from xarray
            
        Returns:
            Dictionary of temporal features
        """
        # Convert to datetime if needed
        if hasattr(time_coord, 'dt'):
            dt = time_coord.dt
        else:
            dt = time_coord
        
        features = {
            'hour': dt.hour.values,
            'day_of_year': dt.dayofyear.values,
            'month': dt.month.values,
            'season': ((dt.month.values - 1) // 3) % 4,  # 0=winter, 1=spring, 2=summer, 3=fall
        }
        
        # Add cyclical encoding for periodic features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365.25)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365.25)
        
        return features
    
    @staticmethod
    def create_temporal_mask(lightning_data: np.ndarray, 
                           min_events_per_timestep: int = 1) -> np.ndarray:
        """
        Create mask for timesteps with sufficient lightning activity.
        
        Args:
            lightning_data: Lightning data (time, lat, lon)
            min_events_per_timestep: Minimum lightning events per timestep
            
        Returns:
            Boolean mask for valid timesteps
        """
        # Count lightning events per timestep
        events_per_timestep = np.sum(lightning_data > 0, axis=(1, 2))
        
        # Create mask
        mask = events_per_timestep >= min_events_per_timestep
        
        return mask
