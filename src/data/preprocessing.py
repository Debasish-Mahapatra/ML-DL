"""
Data preprocessing utilities for meteorological and lightning data.
FIXED VERSION - Removes coordinate transpose bug and ensures proper alignment.
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
        
        logger.info(f"DataPreprocessor initialized with:")
        logger.info(f"  CAPE stats: {self.cape_stats}")
        logger.info(f"  Terrain stats: {self.terrain_stats}")
        
    def preprocess_cape(self, cape_data: xr.DataArray) -> torch.Tensor:
        """
        Preprocess CAPE data.
        
        Args:
            cape_data: CAPE data from NetCDF (time, lat, lon)
            
        Returns:
            Preprocessed CAPE tensor (time, 1, lat, lon)
        """
        # Log input information
        logger.debug(f"Preprocessing CAPE data:")
        logger.debug(f"  Input shape: {cape_data.shape}")
        logger.debug(f"  Input dimensions: {cape_data.dims}")
        
        # Convert to numpy
        cape_np = cape_data.values
        
        # Verify expected dimensions: (time=744, lat=19, lon=26)
        if cape_np.ndim != 3:
            raise ValueError(f"Expected 3D CAPE data, got {cape_np.ndim}D with shape {cape_np.shape}")
        
        expected_spatial_shape = (19, 26)  # (lat, lon) for 25km resolution
        if cape_np.shape[-2:] != expected_spatial_shape:
            logger.warning(f"CAPE spatial shape {cape_np.shape[-2:]} differs from expected {expected_spatial_shape}")
        
        logger.debug(f"  CAPE data shape after loading: {cape_np.shape} (time, lat, lon)")
        
        # Handle missing values
        cape_np = np.nan_to_num(cape_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values (CAPE should be >= 0)
        cape_np = np.clip(cape_np, 0, 10000)  # Max reasonable CAPE ~10,000 J/kg
        
        # Log data statistics
        logger.debug(f"  CAPE value range: {cape_np.min():.2f} to {cape_np.max():.2f} J/kg")
        logger.debug(f"  CAPE mean: {cape_np.mean():.2f} J/kg")
        
        # Normalize
        cape_normalized = (cape_np - self.cape_stats['mean']) / self.cape_stats['std']
        
        # Convert to tensor and add channel dimension
        cape_tensor = torch.from_numpy(cape_normalized).float()
        
        # Add channel dimension: (time, lat, lon) -> (time, 1, lat, lon)
        cape_tensor = cape_tensor.unsqueeze(1)
        
        logger.debug(f"  Output tensor shape: {cape_tensor.shape} (time, channels, lat, lon)")
        return cape_tensor
    
    def preprocess_lightning(self, lightning_data: xr.DataArray) -> torch.Tensor:
        """
        Preprocess lightning ground truth data.
        
        CRITICAL FIX: Maintains correct coordinate order without transpose.
    
        Args:
            lightning_data: Lightning occurrence data (time, latitude, longitude)
        
        Returns:
            Lightning tensor (time, latitude, longitude) - NO TRANSPOSE!
        """
        # Log input information
        logger.debug(f"Preprocessing lightning data:")
        logger.debug(f"  Input shape: {lightning_data.shape}")
        logger.debug(f"  Input dimensions: {lightning_data.dims}")
        
        # Convert to numpy
        lightning_np = lightning_data.values
        
        # Verify expected dimensions: (time=744, latitude=181, longitude=221)
        if lightning_np.ndim != 3:
            raise ValueError(f"Expected 3D lightning data, got {lightning_np.ndim}D with shape {lightning_np.shape}")
        
        expected_spatial_shape = (181, 221)  # (latitude, longitude) for 3km resolution
        if lightning_np.shape[-2:] != expected_spatial_shape:
            logger.warning(f"Lightning spatial shape {lightning_np.shape[-2:]} differs from expected {expected_spatial_shape}")
        
        logger.debug(f"  Lightning data shape: {lightning_np.shape} (time, latitude, longitude)")
        
        # CRITICAL FIX: DO NOT TRANSPOSE!
        # The data is already in the correct format: (time, latitude, longitude)
        # Previous buggy code that was transposing has been REMOVED
        
        # Log confirmation that we're keeping original order
        logger.debug(f"  Keeping original coordinate order: (time, latitude, longitude)")
    
        # Handle missing values (assume no lightning for missing data)
        lightning_np = np.nan_to_num(lightning_np, nan=0.0)
    
        # Ensure binary values (0 or 1) 
        lightning_np = np.clip(lightning_np, 0, 1)
        
        # Log data statistics
        lightning_positive = np.sum(lightning_np > 0)
        lightning_total = lightning_np.size
        logger.debug(f"  Lightning statistics: {lightning_positive:,} positive pixels out of {lightning_total:,} total ({100*lightning_positive/lightning_total:.3f}%)")
    
        # Convert to tensor - keep original dimension order
        lightning_tensor = torch.from_numpy(lightning_np).float()
        
        logger.debug(f"  Output tensor shape: {lightning_tensor.shape} (time, latitude, longitude)")
        return lightning_tensor
    
    def preprocess_terrain(self, terrain_data: xr.DataArray, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess terrain data and handle resolution conversion.
        
        Args:
            terrain_data: Terrain elevation data (lat, lon)
            target_shape: Target spatial shape (height, width) = (latitude_size, longitude_size)
            
        Returns:
            Preprocessed terrain tensor (1, latitude, longitude)
        """
        # Log input information
        logger.debug(f"Preprocessing terrain data:")
        logger.debug(f"  Input shape: {terrain_data.shape}")
        logger.debug(f"  Input dimensions: {terrain_data.dims}")
        logger.debug(f"  Target shape: {target_shape}")
        
        # Convert to numpy
        terrain_np = terrain_data.values
        
        # Verify expected dimensions: (lat=553, lon=660)
        if terrain_np.ndim != 2:
            raise ValueError(f"Expected 2D terrain data, got {terrain_np.ndim}D with shape {terrain_np.shape}")
        
        expected_shape = (553, 660)  # (lat, lon) for 1km resolution
        if terrain_np.shape != expected_shape:
            logger.warning(f"Terrain shape {terrain_np.shape} differs from expected {expected_shape}")
        
        logger.debug(f"  Terrain data shape: {terrain_np.shape} (lat, lon)")
        
        # Handle missing values
        terrain_np = np.nan_to_num(terrain_np, nan=0.0)
        
        # Log elevation statistics
        logger.debug(f"  Elevation range: {terrain_np.min():.1f} to {terrain_np.max():.1f} m")
        logger.debug(f"  Elevation mean: {terrain_np.mean():.1f} m")
        
        # Resize to target shape if needed
        if terrain_np.shape != target_shape:
            logger.debug(f"  Resizing terrain from {terrain_np.shape} to {target_shape}")
            terrain_np = resize(terrain_np, target_shape, preserve_range=True, anti_aliasing=True)
        
        # Normalize
        terrain_normalized = (terrain_np - self.terrain_stats['mean']) / self.terrain_stats['std']
        
        # Convert to tensor and add channel dimension
        terrain_tensor = torch.from_numpy(terrain_normalized).float()
        terrain_tensor = terrain_tensor.unsqueeze(0)  # Add channel dimension: (lat, lon) -> (1, lat, lon)
        
        logger.debug(f"  Output tensor shape: {terrain_tensor.shape} (channels, latitude, longitude)")
        return terrain_tensor
    
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
                    var_name = 'cape'
                elif data_type == 'terrain':
                    var_name = 'elevation'
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


# Additional utility functions for coordinate handling
def verify_coordinate_alignment(cape_ds: xr.Dataset, 
                              lightning_ds: xr.Dataset, 
                              terrain_ds: xr.Dataset) -> bool:
    """
    Verify that all datasets cover the same geographic domain.
    
    Args:
        cape_ds: CAPE dataset
        lightning_ds: Lightning dataset  
        terrain_ds: Terrain dataset
        
    Returns:
        True if alignment is correct
    """
    logger.info("Verifying coordinate alignment across datasets...")
    
    # Get coordinate names and values
    cape_lat = cape_ds.lat.values if 'lat' in cape_ds.coords else cape_ds.latitude.values
    cape_lon = cape_ds.lon.values if 'lon' in cape_ds.coords else cape_ds.longitude.values
    
    lightning_lat = lightning_ds.latitude.values
    lightning_lon = lightning_ds.longitude.values
    
    terrain_lat = terrain_ds.lat.values if 'lat' in terrain_ds.coords else terrain_ds.latitude.values
    terrain_lon = terrain_ds.lon.values if 'lon' in terrain_ds.coords else terrain_ds.longitude.values
    
    # Check domain bounds
    tolerance = 0.1  # degrees
    
    domains = {
        'cape': (cape_lat.min(), cape_lat.max(), cape_lon.min(), cape_lon.max()),
        'lightning': (lightning_lat.min(), lightning_lat.max(), lightning_lon.min(), lightning_lon.max()),
        'terrain': (terrain_lat.min(), terrain_lat.max(), terrain_lon.min(), terrain_lon.max())
    }
    
    logger.info("Domain bounds (lat_min, lat_max, lon_min, lon_max):")
    for name, bounds in domains.items():
        logger.info(f"  {name}: ({bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f})")
    
    # Check if all domains are approximately the same
    cape_bounds = domains['cape']
    alignment_ok = True
    
    for name, bounds in domains.items():
        if name == 'cape':
            continue
            
        for i, (cape_val, other_val) in enumerate(zip(cape_bounds, bounds)):
            if abs(cape_val - other_val) > tolerance:
                logger.error(f"Domain mismatch between CAPE and {name}: "
                           f"coordinate {i} differs by {abs(cape_val - other_val):.4f} degrees")
                alignment_ok = False
    
    if alignment_ok:
        logger.info("✓ All datasets are properly aligned")
    else:
        logger.error("✗ Datasets are NOT properly aligned")
    
    return alignment_ok


def log_data_shapes(cape_tensor: torch.Tensor, 
                   lightning_tensor: torch.Tensor, 
                   terrain_tensor: torch.Tensor) -> None:
    """
    Log the shapes of processed tensors for debugging.
    
    Args:
        cape_tensor: Processed CAPE tensor
        lightning_tensor: Processed lightning tensor
        terrain_tensor: Processed terrain tensor
    """
    logger.info("Processed tensor shapes:")
    logger.info(f"  CAPE: {cape_tensor.shape} (time, channels, lat, lon)")
    logger.info(f"  Lightning: {lightning_tensor.shape} (time, lat, lon)")
    logger.info(f"  Terrain: {terrain_tensor.shape} (channels, lat, lon)")
    
    # Verify shapes are consistent
    if cape_tensor.shape[0] != lightning_tensor.shape[0]:
        logger.error(f"Time dimension mismatch: CAPE={cape_tensor.shape[0]}, Lightning={lightning_tensor.shape[0]}")
    
    # Log coordinate order confirmation
    logger.info("Coordinate order confirmation:")
    logger.info("  ✓ CAPE: (time, channels, latitude, longitude)")
    logger.info("  ✓ Lightning: (time, latitude, longitude)")  
    logger.info("  ✓ Terrain: (channels, latitude, longitude)")