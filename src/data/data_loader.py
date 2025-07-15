"""
PyTorch Lightning DataModule for efficient data loading.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from omegaconf import DictConfig
import xarray as xr

from .dataset import LightningDataset
from .preprocessing import DataPreprocessor
from .augmentation import SpatialAugmentation, MeteorologicalAugmentation
from ..utils.io_utils import DataPathManager

logger = logging.getLogger(__name__)

class LightningDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for lightning prediction.
    Handles data loading, preprocessing, and augmentation.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Lightning DataModule.
        
        Args:
            config: Configuration object containing data settings
        """
        super().__init__()
        self.config = config
        self.data_config = config.data
        
        # Initialize components
        self.path_manager = DataPathManager(self.data_config.root_dir)
        self.preprocessor = None
        
        # Augmentation components
        self.spatial_augmentation = None
        self.meteorological_augmentation = None
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # File pairs for each split
        self.train_files = None
        self.val_files = None
        self.test_files = None
    
    def prepare_data(self):
        """
        Download and prepare data (if needed).
        This is called only once across all processes.
        """
        logger.info("Preparing data...")
        
        # Validate that required directories exist
        root_path = Path(self.data_config.root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Data root directory not found: {root_path}")
        
        # Check for split files
        splits_dir = Path(self.data_config.splits_dir)
        required_splits = ['train_files.txt', 'val_files.txt', 'test_files.txt']
        
        for split_file in required_splits:
            split_path = splits_dir / split_file
            if not split_path.exists():
                logger.warning(f"Split file not found: {split_path}")
                # Could auto-generate splits here if needed
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        This is called on every process.
        """
        logger.info(f"Setting up data for stage: {stage}")
        
        # Load file splits
        self._load_file_splits()
        
        # Initialize preprocessor with normalization stats
        self._setup_preprocessor()
        
        # Initialize augmentations
        self._setup_augmentations()
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            self._setup_train_val_datasets()
        
        if stage == 'test' or stage is None:
            self._setup_test_dataset()
    
    def _load_file_splits(self):
        """Load file pairs for each split."""
        splits_dir = Path(self.data_config.splits_dir)
        
        def load_split_files(split_name: str) -> List[Dict[str, Path]]:
            split_file = splits_dir / f"{split_name}_files.txt"
            
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                return []
            
            file_pairs = []
            
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Expected format: cape_path,lightning_path
                    parts = line.split(',')
                    if len(parts) != 2:
                        logger.warning(f"Invalid line format: {line}")
                        continue
                    
                    cape_rel_path, lightning_rel_path = parts
                    
                    # Convert to absolute paths
                    cape_path = Path(self.data_config.root_dir) / cape_rel_path
                    lightning_path = Path(self.data_config.root_dir) / lightning_rel_path
                    terrain_path = Path(self.data_config.root_dir) / self.data_config.terrain.path
                    
                    # Validate files exist
                    if cape_path.exists() and lightning_path.exists() and terrain_path.exists():
                        file_pairs.append({
                            'cape': cape_path,
                            'lightning': lightning_path,
                            'terrain': terrain_path
                        })
                    else:
                        missing = []
                        if not cape_path.exists():
                            missing.append(str(cape_path))
                        if not lightning_path.exists():
                            missing.append(str(lightning_path))
                        if not terrain_path.exists():
                            missing.append(str(terrain_path))
                        logger.warning(f"Missing files: {missing}")
            
            return file_pairs
        
        self.train_files = load_split_files('train')
        self.val_files = load_split_files('val')
        self.test_files = load_split_files('test')
        
        logger.info(f"Loaded {len(self.train_files)} train, {len(self.val_files)} val, {len(self.test_files)} test file pairs")
    
    def _setup_preprocessor(self):
        """Initialize preprocessor with normalization statistics."""
        logger.info("Setting up data preprocessor...")
        
        # Compute normalization stats from training data
        if self.train_files:
            # Get CAPE files for stats computation
            cape_files = [str(fp['cape']) for fp in self.train_files]
            terrain_files = [str(fp['terrain']) for fp in self.train_files[:1]]  # Terrain is same for all
            
            # Compute stats
            cape_stats = DataPreprocessor().compute_normalization_stats(cape_files, 'cape')
            terrain_stats = DataPreprocessor().compute_normalization_stats(terrain_files, 'terrain')
            
        else:
            logger.warning("No training files found, using default normalization stats")
            cape_stats = {'mean': 0.0, 'std': 1.0}
            terrain_stats = {'mean': 0.0, 'std': 1.0}
        
        self.preprocessor = DataPreprocessor(cape_stats=cape_stats, terrain_stats=terrain_stats)
        
        logger.info(f"CAPE stats: {cape_stats}")
        logger.info(f"Terrain stats: {terrain_stats}")
    
    def _setup_augmentations(self):
        """Initialize augmentation functions."""
        # Spatial augmentation (if enabled in config)
        if hasattr(self.data_config, 'augmentation') and self.data_config.augmentation.spatial:
            spatial_config = self.data_config.augmentation.spatial
            self.spatial_augmentation = SpatialAugmentation(
                rotation_range=spatial_config.get('rotation', [-30, 30]),
                flip_horizontal=spatial_config.get('flip_horizontal', 0.5),
                flip_vertical=spatial_config.get('flip_vertical', 0.5)
            )
        
        # Meteorological augmentation (if enabled in config)
        if hasattr(self.data_config, 'augmentation') and self.data_config.augmentation.meteorological:
            met_config = self.data_config.augmentation.meteorological
            self.meteorological_augmentation = MeteorologicalAugmentation(
                noise_std=met_config.get('noise_std', 0.01),
                scale_factor=met_config.get('scale_factor', [0.95, 1.05])
            )
    
    def _setup_train_val_datasets(self):
        """Setup training and validation datasets."""
        # Training dataset (with augmentation)
        self.train_dataset = LightningDataset(
            file_pairs=self.train_files,
            preprocessor=self.preprocessor,
            sequence_length=self.data_config.temporal.sequence_length,
            spatial_augmentation=self.spatial_augmentation,
            meteorological_augmentation=self.meteorological_augmentation,
            target_cape_shape=tuple(self.data_config.domain.grid_size_25km),     # CHANGED: Native 25km
            target_lightning_shape=tuple(self.data_config.domain.grid_size_3km),
            target_terrain_shape=tuple(self.data_config.domain.grid_size_1km)    # CHANGED: Native 1km
        )
        
        # Validation dataset (no augmentation)
        self.val_dataset = LightningDataset(
            file_pairs=self.val_files,
            preprocessor=self.preprocessor,
            sequence_length=self.data_config.temporal.sequence_length,
            spatial_augmentation=None,
            meteorological_augmentation=None,
            target_cape_shape=tuple(self.data_config.domain.grid_size_25km),     # CHANGED: Native 25km
            target_lightning_shape=tuple(self.data_config.domain.grid_size_3km),
            target_terrain_shape=tuple(self.data_config.domain.grid_size_1km)    # CHANGED: Native 1km
        )
   
    def _setup_test_dataset(self):
        """Setup test dataset."""
        self.test_dataset = LightningDataset(
            file_pairs=self.test_files,
            preprocessor=self.preprocessor,
            sequence_length=self.data_config.temporal.sequence_length,
            spatial_augmentation=None,
            meteorological_augmentation=None,
            target_cape_shape=tuple(self.data_config.domain.grid_size_25km),     # CHANGED: Native 25km
            target_lightning_shape=tuple(self.data_config.domain.grid_size_3km),
            target_terrain_shape=tuple(self.data_config.domain.grid_size_1km)    # CHANGED: Native 1km
        )
   
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            drop_last=True,
            persistent_workers=True if self.data_config.num_workers > 0 else False
        )
   
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            drop_last=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False
        )
   
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            drop_last=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False
        )
   
    def get_sample_batch(self, split: str = 'train') -> Dict:
        """
        Get a sample batch for testing/debugging.
        
        Args:
            split: Which split to sample from ('train', 'val', 'test')
            
        Returns:
            Sample batch dictionary
        """
        if split == 'train' and self.train_dataset:
            return self.train_dataset[0]
        elif split == 'val' and self.val_dataset:
            return self.val_dataset[0]
        elif split == 'test' and self.test_dataset:
            return self.test_dataset[0]
        else:
            raise ValueError(f"Dataset for split '{split}' not available")

class ChunkedDataLoader:
    """
    Custom data loader for handling large files in chunks.
    Useful for very large NetCDF files that don't fit in memory.
    """
   
    def __init__(self, 
                 file_pairs: List[Dict[str, Path]],
                 preprocessor: DataPreprocessor,
                 chunk_size: int = 24,  # hours per chunk
                 batch_size: int = 4,
                 shuffle: bool = True):
        """
        Initialize chunked data loader.
       
        Args:
            file_pairs: List of file dictionaries
            preprocessor: Data preprocessor
            chunk_size: Number of time steps per chunk
            batch_size: Batch size
            shuffle: Whether to shuffle chunks
        """
        self.file_pairs = file_pairs
        self.preprocessor = preprocessor
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
       
        # Build chunk index
        self._build_chunk_index()
   
    def _build_chunk_index(self):
        """Build index of all chunks across all files."""
        self.chunk_index = []
        
        for file_idx, file_dict in enumerate(self.file_pairs):
            try:
                # Get file time dimension
                cape_ds = xr.open_dataset(file_dict['cape'])
                num_times = len(cape_ds.time) if 'time' in cape_ds.dims else 1
                cape_ds.close()
                
                # Create chunks
                for start_time in range(0, num_times, self.chunk_size):
                    end_time = min(start_time + self.chunk_size, num_times)
                    
                    self.chunk_index.append({
                        'file_idx': file_idx,
                        'time_start': start_time,
                        'time_end': end_time
                    })
                   
            except Exception as e:
                logger.warning(f"Error indexing file {file_idx}: {e}")
                continue
       
        logger.info(f"Created {len(self.chunk_index)} chunks")
   
    def __iter__(self):
        """Iterate over chunks."""
        import random
        
        chunk_indices = list(range(len(self.chunk_index)))
        if self.shuffle:
            random.shuffle(chunk_indices)
       
        for chunk_idx in chunk_indices:
            chunk_info = self.chunk_index[chunk_idx]
            file_dict = self.file_pairs[chunk_info['file_idx']]
           
            # Load chunk data
            try:
                chunk_data = self._load_chunk(file_dict, chunk_info)
                
                # Yield batches from chunk
                for batch in self._create_batches(chunk_data):
                    yield batch
                   
            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_idx}: {e}")
                continue
   
    def _load_chunk(self, file_dict: Dict[str, Path], chunk_info: Dict) -> Dict:
        """Load a single chunk of data."""
        # Implementation for loading temporal chunks
        # This would be used for very large files
        pass
   
    def _create_batches(self, chunk_data: Dict) -> List[Dict]:
        """Create batches from chunk data."""
        # Implementation for creating batches from loaded chunk
        pass