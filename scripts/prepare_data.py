"""
Data preparation script for lightning prediction.
Creates train/validation/test splits and preprocesses data.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from omegaconf import OmegaConf

from src.utils.config import get_config
from src.utils.io_utils import DataPathManager, NetCDFHandler
from src.data.preprocessing import DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data_files(data_dir: str) -> dict:
    """Validate and inventory data files."""
    
    path_manager = DataPathManager(data_dir)
    
    # Get all CAPE files
    cape_files = path_manager.get_file_list('cape')
    lightning_files = path_manager.get_file_list('lightning')
    terrain_files = path_manager.get_file_list('terrain')
    
    logger.info(f"Found {len(cape_files)} CAPE files")
    logger.info(f"Found {len(lightning_files)} lightning files")
    logger.info(f"Found {len(terrain_files)} terrain files")
    
    # Validate terrain file
    if len(terrain_files) != 1:
        raise ValueError(f"Expected exactly 1 terrain file, found {len(terrain_files)}")
    
    # Match CAPE and lightning files
    valid_pairs = []
    cape_dict = {f.stem: f for f in cape_files}
    lightning_dict = {f.stem.replace('lightning_', 'cape_'): f for f in lightning_files}
    
    for cape_key, cape_file in cape_dict.items():
        if cape_key in lightning_dict:
            lightning_file = lightning_dict[cape_key]
            
            # Validate file structure
            if validate_file_pair(cape_file, lightning_file, terrain_files[0]):
                valid_pairs.append({
                    'cape': cape_file,
                    'lightning': lightning_file,
                    'terrain': terrain_files[0]
                })
            else:
                logger.warning(f"Skipping invalid file pair: {cape_file.name}")
        else:
            logger.warning(f"No matching lightning file for {cape_file.name}")
    
    logger.info(f"Found {len(valid_pairs)} valid file pairs")
    
    return {
        'valid_pairs': valid_pairs,
        'terrain_file': terrain_files[0]
    }

def validate_file_pair(cape_file: Path, lightning_file: Path, terrain_file: Path) -> bool:
    """Validate that a file pair has compatible structure."""
    
    try:
        # Check CAPE file
        cape_info = NetCDFHandler.get_file_info(cape_file)
        if not cape_info or 'cape' not in cape_info.get('variables', []):
            logger.warning(f"CAPE file missing 'cape' variable: {cape_file}")
            return False
        
        # Check lightning file
        lightning_info = NetCDFHandler.get_file_info(lightning_file)
        if not lightning_info or 'lightning_occurrence' not in lightning_info.get('variables', []):
            logger.warning(f"Lightning file missing 'lightning_occurrence' variable: {lightning_file}")
            return False
        
        # Check terrain file
        terrain_info = NetCDFHandler.get_file_info(terrain_file)
        if not terrain_info or 'elevation' not in terrain_info.get('variables', []):
            logger.warning(f"Terrain file missing 'elevation' variable: {terrain_file}")
            return False
        
        # Basic dimension checks
        cape_dims = cape_info.get('dimensions', {})
        lightning_dims = lightning_info.get('dimensions', {})
        
        # Check that time dimensions are reasonable
        cape_time = cape_dims.get('time', 0)
        lightning_time = lightning_dims.get('time', 0)
        
        if abs(cape_time - lightning_time) > 1:  # Allow small differences
            logger.warning(f"Time dimension mismatch: CAPE={cape_time}, Lightning={lightning_time}")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating file pair: {e}")
        return False

def extract_year_month_from_filename(filename: str) -> tuple:
    """Extract year and month from filename like 'cape_2020_01.nc'."""
    try:
        # Remove extension and split by underscore
        name_parts = filename.split('.')[0].split('_')
        
        # Expected format: variable_year_month
        if len(name_parts) >= 3:
            year = int(name_parts[-2])
            month = int(name_parts[-1])
            return year, month
        else:
            raise ValueError(f"Unexpected filename format: {filename}")
            
    except Exception as e:
        logger.warning(f"Could not extract year/month from {filename}: {e}")
        return None, None

def get_season_from_month(month: int) -> str:
    """Get season from month number."""
    if month in [12, 1, 2]:
        return 'DJF'  # Winter
    elif month in [3, 4, 5]:
        return 'MAM'  # Spring/Pre-monsoon
    elif month in [6, 7, 8]:
        return 'JJA'  # Summer/Monsoon
    elif month in [9, 10, 11]:
        return 'SON'  # Autumn/Post-monsoon
    else:
        raise ValueError(f"Invalid month: {month}")

def create_seasonal_aware_splits(file_pairs: list, 
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                years_for_training: list = [2019, 2020, 2021],
                                random_seed: int = 42) -> dict:
    """
    Create seasonal-aware splits ensuring each season is represented in all splits.
    
    Args:
        file_pairs: List of file pair dictionaries
        train_ratio: Training data ratio
        val_ratio: Validation data ratio  
        test_ratio: Test data ratio
        years_for_training: Years to use for training data
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with train/val/test splits
    """
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    
    # Group files by season and year
    seasonal_files = {
        'DJF': [],  # Dec, Jan, Feb
        'MAM': [],  # Mar, Apr, May
        'JJA': [],  # Jun, Jul, Aug
        'SON': []   # Sep, Oct, Nov
    }
    
    # Separate training years from others
    training_pairs = []
    other_pairs = []
    
    for pair in file_pairs:
        cape_filename = pair['cape'].name
        year, month = extract_year_month_from_filename(cape_filename)
        
        if year is None or month is None:
            logger.warning(f"Skipping file with unparseable name: {cape_filename}")
            continue
        
        if year in years_for_training:
            training_pairs.append((pair, year, month))
        else:
            other_pairs.append((pair, year, month))
    
    logger.info(f"Found {len(training_pairs)} files from training years {years_for_training}")
    logger.info(f"Found {len(other_pairs)} files from other years")
    
    # Group training files by season
    for pair, year, month in training_pairs:
        season = get_season_from_month(month)
        seasonal_files[season].append(pair)
    
    # Create splits for each season
    train_files = []
    val_files = []
    test_files = []
    
    for season, season_files in seasonal_files.items():
        if not season_files:
            logger.warning(f"No files found for season {season}")
            continue
        
        # Shuffle files within season
        random.shuffle(season_files)
        
        # Calculate split indices
        n_total = len(season_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split files
        season_train = season_files[:n_train]
        season_val = season_files[n_train:n_train + n_val]
        season_test = season_files[n_train + n_val:]
        
        # Add to overall splits
        train_files.extend(season_train)
        val_files.extend(season_val)
        test_files.extend(season_test)
        
        logger.info(f"Season {season}: {len(season_train)} train, {len(season_val)} val, {len(season_test)} test files")
    
    # Shuffle the combined splits to mix seasons
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    logger.info(f"Total seasonal-aware split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Log seasonal distribution
    for split_name, split_files in [("Train", train_files), ("Val", val_files), ("Test", test_files)]:
        season_counts = {'DJF': 0, 'MAM': 0, 'JJA': 0, 'SON': 0}
        for pair in split_files:
            year, month = extract_year_month_from_filename(pair['cape'].name)
            if year and month:
                season = get_season_from_month(month)
                season_counts[season] += 1
        
        logger.info(f"{split_name} seasonal distribution: {season_counts}")
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'other_years': other_pairs  # Files from years not used for training (e.g., 2022)
    }

def create_data_splits(file_pairs: list, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      split_strategy: str = "seasonal_aware",
                      years_for_training: list = [2019, 2020, 2021],
                      random_seed: int = 42) -> dict:
    """Create train/validation/test splits using specified strategy."""
    
    if split_strategy == "seasonal_aware":
        return create_seasonal_aware_splits(
            file_pairs, train_ratio, val_ratio, test_ratio, 
            years_for_training, random_seed
        )
    elif split_strategy == "temporal":
        # Sort by filename (assumes chronological naming)
        sorted_pairs = sorted(file_pairs, key=lambda x: x['cape'].name)
        
        n_total = len(sorted_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_pairs = sorted_pairs[:n_train]
        val_pairs = sorted_pairs[n_train:n_train + n_val]
        test_pairs = sorted_pairs[n_train + n_val:]
        
        logger.info(f"Temporal split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    elif split_strategy == "random":
        # Random split
        random.seed(random_seed)
        random.shuffle(file_pairs)
        
        n_total = len(file_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_pairs = file_pairs[:n_train]
        val_pairs = file_pairs[n_train:n_train + n_val]
        test_pairs = file_pairs[n_train + n_val:]
        
        logger.info(f"Random split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

def save_split_files(splits: dict, output_dir: str, data_root: str):
    """Save split files for training."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_root_path = Path(data_root)
    
    for split_name, file_pairs in splits.items():
        if split_name == 'other_years':  # Skip other years in main splits
            continue
            
        split_file = output_path / f"{split_name}_files.txt"
        
        with open(split_file, 'w') as f:
            f.write("# Lightning prediction data splits\n")
            f.write(f"# Split: {split_name}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# Format: cape_path,lightning_path\n")
            
            for pair in file_pairs:
                # Convert to relative paths
                cape_rel = pair['cape'].relative_to(data_root_path)
                lightning_rel = pair['lightning'].relative_to(data_root_path)
                
                f.write(f"{cape_rel},{lightning_rel}\n")
        
        logger.info(f"Saved {len(file_pairs)} entries to {split_file}")
    
    # Save other years separately for future use
    if 'other_years' in splits and splits['other_years']:
        other_years_file = output_path / "other_years_files.txt"
        
        with open(other_years_file, 'w') as f:
            f.write("# Files from years not used in training (e.g., 2022 for final evaluation)\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# Format: cape_path,lightning_path,year,month\n")
            
            for pair_info in splits['other_years']:
                pair, year, month = pair_info
                cape_rel = pair['cape'].relative_to(data_root_path)
                lightning_rel = pair['lightning'].relative_to(data_root_path)
                
                f.write(f"{cape_rel},{lightning_rel},{year},{month}\n")
        
        logger.info(f"Saved {len(splits['other_years'])} other year entries to {other_years_file}")

def compute_dataset_statistics(file_pairs: list, terrain_file: Path) -> dict:
    """Compute dataset statistics for normalization."""
    
    logger.info("Computing dataset statistics...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Compute CAPE statistics
    cape_files = [str(pair['cape']) for pair in file_pairs]
    cape_stats = preprocessor.compute_normalization_stats(cape_files, 'cape')
    
    # Compute terrain statistics
    terrain_stats = preprocessor.compute_normalization_stats([str(terrain_file)], 'terrain')
    
    # Compute lightning statistics
    lightning_files = [str(pair['lightning']) for pair in file_pairs]
    lightning_events = []
    
    for lightning_file in lightning_files[:10]:  # Sample subset for efficiency
        try:
            ds = NetCDFHandler.load_netcdf(lightning_file, variables=['lightning_occurrence'])
            if ds is not None:
                data = ds['lightning_occurrence'].values
                lightning_events.extend(data.flatten())
                ds.close()
        except Exception as e:
            logger.warning(f"Error processing {lightning_file}: {e}")
    
    if lightning_events:
        lightning_events = np.array(lightning_events)
        lightning_stats = {
            'total_events': int(np.sum(lightning_events)),
            'event_rate': float(np.mean(lightning_events)),
            'total_samples': len(lightning_events)
        }
    else:
        lightning_stats = {'total_events': 0, 'event_rate': 0.0, 'total_samples': 0}
    
    stats = {
        'cape': cape_stats,
        'terrain': terrain_stats,
        'lightning': lightning_stats,
        'computed_at': datetime.now().isoformat()
    }
    
    logger.info(f"Dataset statistics computed:")
    logger.info(f"  CAPE: mean={cape_stats['mean']:.2f}, std={cape_stats['std']:.2f}")
    logger.info(f"  Terrain: mean={terrain_stats['mean']:.2f}, std={terrain_stats['std']:.2f}")
    logger.info(f"  Lightning: {lightning_stats['total_events']} events, "
               f"rate={lightning_stats['event_rate']:.6f}")
    
    return stats

def main():
    """Main data preparation function."""
    
    parser = argparse.ArgumentParser(description="Prepare Lightning Prediction Data")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Root directory containing processed data")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                       help="Output directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation data ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test data ratio")
    parser.add_argument("--split-strategy", type=str, default="seasonal_aware",
                       choices=["temporal", "random", "seasonal_aware"],
                       help="Data splitting strategy")
    parser.add_argument("--training-years", nargs="+", type=int, 
                       default=[2019, 2020, 2021],
                       help="Years to use for training data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    parser.add_argument("--compute-stats", action="store_true",
                       help="Compute and save dataset statistics")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate data without creating splits")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1
    
    # Validate and inventory data files
    try:
        logger.info(f"Validating data in {args.data_dir}")
        data_inventory = validate_data_files(args.data_dir)
        
        if len(data_inventory['valid_pairs']) == 0:
            logger.error("No valid file pairs found")
            return 1
            
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return 1
    
    # Stop here if validation only
    if args.validate_only:
        logger.info("Data validation completed successfully")
        return 0
    
    # Create data splits
    try:
        logger.info("Creating data splits...")
        splits = create_data_splits(
            data_inventory['valid_pairs'],
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.split_strategy,
            args.training_years,
            args.seed
        )
        
    except Exception as e:
        logger.error(f"Failed to create data splits: {e}")
        return 1
    
    # Save split files
    try:
        logger.info("Saving split files...")
        save_split_files(splits, args.output_dir, args.data_dir)
        
    except Exception as e:
        logger.error(f"Failed to save split files: {e}")
        return 1
    
    # Compute dataset statistics
    if args.compute_stats:
        try:
            stats = compute_dataset_statistics(
                splits['train'], 
                data_inventory['terrain_file']
            )
            
            # Save statistics
            stats_file = Path(args.output_dir) / "dataset_statistics.yaml"
            OmegaConf.save(OmegaConf.create(stats), stats_file)
            logger.info(f"Saved dataset statistics to {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to compute statistics: {e}")
            return 1
    
    # Create summary
    summary = {
        'data_preparation': {
            'input_directory': args.data_dir,
            'output_directory': args.output_dir,
            'total_file_pairs': len(data_inventory['valid_pairs']),
            'train_files': len(splits['train']),
            'val_files': len(splits['val']),
            'test_files': len(splits['test']),
            'other_years_files': len(splits.get('other_years', [])),
            'split_strategy': args.split_strategy,
            'training_years': args.training_years,
            'split_ratios': {
                'train': args.train_ratio,
                'val': args.val_ratio,
                'test': args.test_ratio
            },
            'created_at': datetime.now().isoformat(),
            'random_seed': args.seed
        }
    }
    
    summary_file = Path(args.output_dir) / "preparation_summary.yaml"
    OmegaConf.save(OmegaConf.create(summary), summary_file)
    
    logger.info(f"Data preparation completed successfully")
    logger.info(f"Summary saved to {summary_file}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)