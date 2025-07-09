"""
Preprocessing script to crop and split raw data for lightning prediction.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def crop_to_domain(ds, lon_min=81.1644, lon_max=87.52883, lat_min=17.76351, lat_max=22.62838):
    """
    Crop dataset to Odisha domain with robust coordinate handling.
    """
    
    # Find longitude and latitude coordinate names
    lon_names = ['longitude', 'lon', 'long', 'x']
    lat_names = ['latitude', 'lat', 'y']
    
    lon_coord = None
    lat_coord = None
    
    for name in lon_names:
        if name in ds.coords:
            lon_coord = name
            break
    
    for name in lat_names:
        if name in ds.coords:
            lat_coord = name
            break
    
    if lon_coord is None or lat_coord is None:
        raise ValueError(f"Could not find lat/lon coordinates. Available coords: {list(ds.coords.keys())}")
    
    logger.info(f"Using coordinates: {lat_coord}, {lon_coord}")
    
    # Get coordinate values
    lon_vals = ds.coords[lon_coord].values
    lat_vals = ds.coords[lat_coord].values
    
    # Print coordinate ranges for debugging
    logger.info(f"Data coordinate ranges:")
    logger.info(f"  {lon_coord}: {lon_vals.min():.4f} to {lon_vals.max():.4f}")
    logger.info(f"  {lat_coord}: {lat_vals.min():.4f} to {lat_vals.max():.4f}")
    logger.info(f"Target domain: lon {lon_min:.4f} to {lon_max:.4f}, lat {lat_min:.4f} to {lat_max:.4f}")
    
    # Handle longitude wrapping (0-360 vs -180-180)
    if lon_vals.max() > 180 and lon_min < 180:
        # Data is 0-360, domain is -180-180, convert domain
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360
        logger.info(f"Converted longitude bounds to: {lon_min:.4f} to {lon_max:.4f}")
    
    # Check if latitude is inverted (decreasing order)
    lat_increasing = lat_vals[0] < lat_vals[-1]
    
    if not lat_increasing:
        logger.info("Latitude coordinates are decreasing - will handle appropriately")
        # For decreasing latitude, we need to swap min/max for slicing
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    
    # Crop to domain
    try:
        ds_cropped = ds.sel(
            {lon_coord: slice(lon_min, lon_max),
             lat_coord: lat_slice}
        )
    except Exception as e:
        logger.error(f"Error cropping with slice: {e}")
        # Try with nearest neighbor selection
        lon_mask = (ds.coords[lon_coord] >= lon_min) & (ds.coords[lon_coord] <= lon_max)
        lat_mask = (ds.coords[lat_coord] >= lat_min) & (ds.coords[lat_coord] <= lat_max)
        
        ds_cropped = ds.sel(
            {lon_coord: ds.coords[lon_coord][lon_mask],
             lat_coord: ds.coords[lat_coord][lat_mask]}
        )
    
    logger.info(f"Cropped from {ds.sizes} to {ds_cropped.sizes}")
    
    # Check if cropping was successful
    if any(dim_size == 0 for dim_size in ds_cropped.sizes.values()):
        logger.error(f"Cropping resulted in empty dataset: {ds_cropped.sizes}")
        logger.error("This usually means the domain is outside the data bounds")
        
        # Try to find the closest available region
        logger.info("Available latitude range: {:.4f} to {:.4f}".format(lat_vals.min(), lat_vals.max()))
        logger.info("Available longitude range: {:.4f} to {:.4f}".format(lon_vals.min(), lon_vals.max()))
        
        raise ValueError(f"Cropping resulted in empty dataset: {ds_cropped.sizes}")
    
    return ds_cropped

def process_lightning_data(input_dir, output_dir):
    """Process lightning data files."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir) / "lightning"
    
    # Lightning files
    lightning_files = list(input_path.glob("*lightning_3km_hourly_grid.nc"))
    
    if not lightning_files:
        raise FileNotFoundError(f"No lightning files found in {input_dir}")
    
    logger.info(f"Found {len(lightning_files)} lightning files")
    
    for lightning_file in lightning_files:
        logger.info(f"Processing {lightning_file.name}")
        
        # Extract year from filename
        year = lightning_file.name.split('_')[0]
        
        # Load dataset
        ds = xr.open_dataset(lightning_file)
        
        # Crop to domain (lightning data is already at 3km for Odisha, but let's be safe)
        ds_cropped = crop_to_domain(ds)
        
        # Check for lightning variable
        lightning_vars = ['lightning', 'lightning_occurrence', 'flash_count', 'flash']
        lightning_var = None
        
        for var in lightning_vars:
            if var in ds_cropped.data_vars:
                lightning_var = var
                break
        
        if lightning_var is None:
            logger.warning(f"No lightning variable found in {lightning_file}. Variables: {list(ds_cropped.data_vars.keys())}")
            ds.close()
            continue
        
        logger.info(f"Using lightning variable: {lightning_var}")
        
        # Split by month
        for month in range(1, 13):
            try:
                # Select month data
                ds_month = ds_cropped.sel(time=ds_cropped.time.dt.month == month)
                
                if len(ds_month.time) == 0:
                    logger.warning(f"No data for {year}-{month:02d}")
                    continue
                
                # Create output directory
                year_output_dir = output_path / year
                year_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Rename variable to standard name
                if lightning_var != 'lightning_occurrence':
                    ds_month = ds_month.rename({lightning_var: 'lightning_occurrence'})
                
                # Save monthly file
                output_file = year_output_dir / f"lightning_{year}_{month:02d}.nc"
                
                # Add metadata
                ds_month.attrs.update({
                    'title': 'Lightning occurrence data for Odisha',
                    'source': f'Processed from {lightning_file.name}',
                    'processing_date': datetime.now().isoformat(),
                    'domain': 'Odisha (81.16-87.53E, 17.76-22.63N)',
                    'resolution': '3km'
                })
                
                ds_month.to_netcdf(output_file)
                logger.info(f"Saved {output_file} with {len(ds_month.time)} time steps")
                
            except Exception as e:
                logger.error(f"Error processing {year}-{month:02d}: {e}")
                continue
        
        ds.close()

def process_cape_data(cape_file, output_dir):
    """Process CAPE data file."""
    
    output_path = Path(output_dir) / "meteorological" / "cape"
    
    logger.info(f"Processing CAPE data from {cape_file}")
    
    # Load CAPE dataset
    ds = xr.open_dataset(cape_file)
    
    logger.info(f"Original CAPE dataset: {ds.sizes}")
    logger.info(f"Variables: {list(ds.data_vars.keys())}")
    
    # Crop to Odisha domain
    ds_cropped = crop_to_domain(ds)
    
    # Find CAPE variable
    cape_vars = ['cape', 'CAPE', 'convective_available_potential_energy']
    cape_var = None
    
    for var in cape_vars:
        if var in ds_cropped.data_vars:
            cape_var = var
            break
    
    if cape_var is None:
        raise ValueError(f"No CAPE variable found. Variables: {list(ds_cropped.data_vars.keys())}")
    
    logger.info(f"Using CAPE variable: {cape_var}")
    
    # Get time range
    time_range = ds_cropped.time
    years = np.unique(time_range.dt.year.values)
    
    logger.info(f"Processing years: {years}")
    
    # Split by year and month
    for year in years:
        logger.info(f"Processing year {year}")
        
        # Select year data
        ds_year = ds_cropped.sel(time=ds_cropped.time.dt.year == year)
        
        for month in range(1, 13):
            try:
                # Select month data
                ds_month = ds_year.sel(time=ds_year.time.dt.month == month)
                
                if len(ds_month.time) == 0:
                    logger.warning(f"No CAPE data for {year}-{month:02d}")
                    continue
                
                # Create output directory
                year_output_dir = output_path / str(year)
                year_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Rename variable to standard name
                if cape_var != 'cape':
                    ds_month = ds_month.rename({cape_var: 'cape'})
                
                # Keep only CAPE variable and coordinates
                ds_month = ds_month[['cape']]
                
                # Save monthly file
                output_file = year_output_dir / f"cape_{year}_{month:02d}.nc"
                
                # Add metadata
                ds_month.attrs.update({
                    'title': 'CAPE data for Odisha',
                    'source': f'Processed from {cape_file}',
                    'processing_date': datetime.now().isoformat(),
                    'domain': 'Odisha (81.16-87.53E, 17.76-22.63N)',
                    'resolution': '25km (approximate)'
                })
                
                ds_month.to_netcdf(output_file)
                logger.info(f"Saved {output_file} with {len(ds_month.time)} time steps")
                
            except Exception as e:
                logger.error(f"Error processing CAPE {year}-{month:02d}: {e}")
                continue
    
    ds.close()

def process_terrain_data(terrain_file, output_dir, reference_lightning_file=None):
    """Process terrain data."""
    
    output_path = Path(output_dir) / "terrain"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing terrain data from {terrain_file}")
    
    # Load terrain dataset
    ds = xr.open_dataset(terrain_file)
    
    logger.info(f"Original terrain dataset: {ds.sizes}")
    logger.info(f"Variables: {list(ds.data_vars.keys())}")
    
    # Crop to Odisha domain
    ds_cropped = crop_to_domain(ds)
    
    # Find elevation variable
    elev_vars = ['elevation', 'height', 'z', 'topo', 'dem']
    elev_var = None
    
    for var in elev_vars:
        if var in ds_cropped.data_vars:
            elev_var = var
            break
    
    if elev_var is None:
        raise ValueError(f"No elevation variable found. Variables: {list(ds_cropped.data_vars.keys())}")
    
    logger.info(f"Using elevation variable: {elev_var}")
    
    # Rename to standard name
    if elev_var != 'elevation':
        ds_cropped = ds_cropped.rename({elev_var: 'elevation'})
    
    # Keep only elevation variable
    ds_cropped = ds_cropped[['elevation']]
    
    # If reference lightning file provided, interpolate to same grid
    if reference_lightning_file:
        logger.info(f"Interpolating terrain to match lightning grid from {reference_lightning_file}")
        
        ref_ds = xr.open_dataset(reference_lightning_file)
        ref_ds_cropped = crop_to_domain(ref_ds)
        
        # Interpolate terrain to lightning grid
        ds_cropped = ds_cropped.interp_like(ref_ds_cropped, method='linear')
        
        ref_ds.close()
    
    # Save terrain file
    output_file = output_path / "terrain_odisha_1km.nc"
    
    # Add metadata
    ds_cropped.attrs.update({
        'title': 'Terrain elevation data for Odisha',
        'source': f'Processed from {terrain_file}',
        'processing_date': datetime.now().isoformat(),
        'domain': 'Odisha (81.16-87.53E, 17.76-22.63N)',
        'resolution': '1km (approximate)'
    })
    
    ds_cropped.to_netcdf(output_file)
    logger.info(f"Saved terrain data to {output_file}")
    
    ds.close()

def main():
    """Main preprocessing function."""
    
    parser = argparse.ArgumentParser(description="Preprocess Lightning Prediction Raw Data")
    parser.add_argument("--lightning-dir", type=str, required=True,
                       help="Directory containing lightning NC files")
    parser.add_argument("--cape-file", type=str, required=True,
                       help="CAPE NetCDF file")
    parser.add_argument("--terrain-file", type=str, required=True,
                       help="Terrain NetCDF file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--domain", nargs=4, type=float, 
                       default=[81.1644, 87.52883, 17.76351, 22.62838],
                       help="Domain bounds: lon_min lon_max lat_min lat_max")
    
    args = parser.parse_args()
    
    # Validate input files
    lightning_dir = Path(args.lightning_dir)
    if not lightning_dir.exists():
        logger.error(f"Lightning directory not found: {lightning_dir}")
        return 1
    
    cape_file = Path(args.cape_file)
    if not cape_file.exists():
        logger.error(f"CAPE file not found: {cape_file}")
        return 1
    
    terrain_file = Path(args.terrain_file)
    if not terrain_file.exists():
        logger.error(f"Terrain file not found: {terrain_file}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting preprocessing...")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Process lightning data
        logger.info("Processing lightning data...")
        process_lightning_data(lightning_dir, output_dir)
        
        # Process CAPE data
        logger.info("Processing CAPE data...")
        process_cape_data(cape_file, output_dir)
        
        # Process terrain data
        logger.info("Processing terrain data...")
        # Use first lightning file as reference for grid interpolation
        lightning_files = list(lightning_dir.glob("*lightning*.nc"))
        reference_file = lightning_files[0] if lightning_files else None
        
        process_terrain_data(terrain_file, output_dir, reference_file)
        
        logger.info("Preprocessing completed successfully!")
        
        # Print summary
        logger.info("\nProcessed data structure:")
        output_path = Path(output_dir)
        for root_path in output_path.rglob("*"):
            if root_path.is_dir() and root_path != output_path:
                nc_files = list(root_path.glob("*.nc"))
                if nc_files:
                    relative_path = root_path.relative_to(output_path)
                    logger.info(f"  {relative_path}: {len(nc_files)} files")
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
