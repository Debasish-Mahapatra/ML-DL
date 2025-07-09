"""
Input/Output utilities for handling NetCDF files.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class NetCDFHandler:
    """Handles NetCDF file operations for meteorological data."""
    
    @staticmethod
    def load_netcdf(file_path: Union[str, Path], variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Load NetCDF file with optional variable selection.
        
        Args:
            file_path: Path to NetCDF file
            variables: List of variables to load (if None, load all)
            
        Returns:
            xarray Dataset
        """
        try:
            ds = xr.open_dataset(file_path)
            
            if variables:
                # Select only requested variables
                available_vars = list(ds.data_vars.keys())
                missing_vars = set(variables) - set(available_vars)
                
                if missing_vars:
                    logger.warning(f"Variables not found in {file_path}: {missing_vars}")
                    variables = [v for v in variables if v in available_vars]
                
                if variables:
                    ds = ds[variables]
                else:
                    logger.error(f"No requested variables found in {file_path}")
                    return None
                    
            return ds
            
        except Exception as e:
            logger.error(f"Error loading NetCDF file {file_path}: {e}")
            return None
    
    @staticmethod
    def validate_netcdf_structure(file_path: Union[str, Path], 
                                expected_dims: Dict[str, int],
                                expected_vars: List[str]) -> bool:
        """
        Validate NetCDF file structure.
        
        Args:
            file_path: Path to NetCDF file
            expected_dims: Expected dimensions {dim_name: size}
            expected_vars: Expected variable names
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ds = xr.open_dataset(file_path)
            
            # Check dimensions
            for dim_name, expected_size in expected_dims.items():
                if dim_name not in ds.dims:
                    logger.error(f"Missing dimension '{dim_name}' in {file_path}")
                    return False
                    
                actual_size = ds.dims[dim_name]
                if actual_size != expected_size:
                    logger.warning(
                        f"Dimension '{dim_name}' size mismatch in {file_path}: "
                        f"expected {expected_size}, got {actual_size}"
                    )
            
            # Check variables
            available_vars = list(ds.data_vars.keys())
            missing_vars = set(expected_vars) - set(available_vars)
            
            if missing_vars:
                logger.error(f"Missing variables in {file_path}: {missing_vars}")
                return False
                
            ds.close()
            return True
            
        except Exception as e:
            logger.error(f"Error validating NetCDF file {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict:
        """Get information about NetCDF file."""
        try:
            ds = xr.open_dataset(file_path)
            
            info = {
                'dimensions': dict(ds.dims),
                'variables': list(ds.data_vars.keys()),
                'coordinates': list(ds.coords.keys()),
                'attributes': dict(ds.attrs),
                'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024)
            }
            
            ds.close()
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}

class DataPathManager:
    """Manages data file paths and organization."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        
    def get_file_list(self, data_type: str, year: Optional[int] = None) -> List[Path]:
        """
        Get list of files for a specific data type.
        
        Args:
            data_type: Type of data ('cape', 'lightning', 'era5')
            year: Specific year to filter (if None, get all years)
            
        Returns:
            List of file paths
        """
        if data_type == 'terrain':
            # Terrain is a single file
            return [self.root_dir / "terrain" / "terrain_odisha_1km.nc"]
        
        # For meteorological and lightning data
        base_path = self.root_dir / ("meteorological" if data_type in ['cape', 'era5'] else data_type)
        
        if data_type in ['cape', 'era5']:
            base_path = base_path / data_type
            
        files = []
        
        if year:
            year_path = base_path / str(year)
            if year_path.exists():
                files.extend(sorted(year_path.glob(f"{data_type}_*.nc")))
        else:
            # Get all years
            for year_dir in sorted(base_path.glob("20*")):
                if year_dir.is_dir():
                    files.extend(sorted(year_dir.glob(f"{data_type}_*.nc")))
                    
        return files
    
    def get_matching_files(self, cape_file: Path) -> Dict[str, Path]:
        """
        Get corresponding lightning and terrain files for a CAPE file.
        
        Args:
            cape_file: Path to CAPE file
            
        Returns:
            Dictionary with paths to matching files
        """
        # Extract year and month from CAPE filename
        # Expected format: cape_YYYY_MM.nc
        parts = cape_file.stem.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid CAPE filename format: {cape_file.name}")
            
        year, month = parts[1], parts[2]
        
        # Find corresponding lightning file
        lightning_file = self.root_dir / "lightning" / year / f"lightning_{year}_{month}.nc"
        
        # Terrain file (always the same)
        terrain_file = self.root_dir / "terrain" / "terrain_odisha_1km.nc"
        
        return {
            'cape': cape_file,
            'lightning': lightning_file,
            'terrain': terrain_file
        }
