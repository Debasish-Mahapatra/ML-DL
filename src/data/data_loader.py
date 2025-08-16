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

# MEMORY TRACKING IMPORTS - NEW ADDITION
from ..utils.memory_tracker import memory_checkpoint, trace_memory_line, MemoryContext

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
    
    @memory_checkpoint("DATAMODULE_SETUP")
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        This is called on every process.
        """
        logger.info(f"Setting up data for stage: {stage}")
        
        trace_memory_line()  # Start of setup
        
        # Load file splits
        with MemoryContext("LOAD_FILE_SPLITS"):
            self._load_file_splits()
        trace_memory_line()  # After file splits loading
        
        # Initialize preprocessor with normalization stats
        with MemoryContext("SETUP_PREPROCESSOR"):
            self._setup_preprocessor()
        trace_memory_line()  # After preprocessor setup
        
        # Initialize augmentations
        with MemoryContext("SETUP_AUGMENTATIONS"):
            self._setup_augmentations()
        trace_memory_line()  # After augmentation setup
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            with MemoryContext("SETUP_TRAIN_VAL_DATASETS"):
                self._setup_train_val_datasets()
            trace_memory_line()  # After train/val dataset setup
        
        if stage == 'test' or stage is None:
            with MemoryContext("SETUP_TEST_DATASET"):
                self._setup_test_dataset()
            trace_memory_line()  # After test dataset setup
    
    def _load_file_splits(self):
        """Load file pairs for each split."""
        
        splits_dir = Path(self.data_config.splits_dir)
        
        # Load training files
        train_file = splits_dir / 'train_files.txt'
        if train_file.exists():
            self.train_files = self._parse_file_list(train_file)
            logger.info(f"Loaded {len(self.train_files)} training file pairs")
        else:
            raise FileNotFoundError(f"Training split file not found: {train_file}")
        
        # Load validation files
        val_file = splits_dir / 'val_files.txt'
        if val_file.exists():
            self.val_files = self._parse_file_list(val_file)
            logger.info(f"Loaded {len(self.val_files)} validation file pairs")
        else:
            raise FileNotFoundError(f"Validation split file not found: {val_file}")
        
        # Load test files (optional)
        test_file = splits_dir / 'test_files.txt'
        if test_file.exists():
            self.test_files = self._parse_file_list(test_file)
            logger.info(f"Loaded {len(self.test_files)} test file pairs")
        else:
            logger.warning("Test split file not found - test evaluation will be skipped")
            self.test_files = []
    
    def _parse_file_list(self, file_path: Path) -> List[Dict[str, Path]]:
        """Parse file list from text file."""
    
        file_pairs = []
    
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
            
                # Parse file paths (format: cape_path,lightning_path)
                parts = line.split(',')
                if len(parts) >= 2:
                    cape_rel_path = parts[0].strip()
                    lightning_rel_path = parts[1].strip()
                
                    # Convert to absolute paths
                    cape_file = Path(self.data_config.root_dir) / cape_rel_path
                    lightning_file = Path(self.data_config.root_dir) / lightning_rel_path
                    terrain_file = Path(self.data_config.root_dir) / "terrain" / "terrain_odisha_1km.nc"
                
                    # Validate files exist
                    if cape_file.exists() and lightning_file.exists() and terrain_file.exists():
                        file_pair = {
                            'cape': cape_file,
                            'lightning': lightning_file,
                            'terrain': terrain_file
                        }
                    
                        # Add ERA5 file if provided
                        if len(parts) >= 3:
                            era5_rel_path = parts[2].strip()
                            era5_file = Path(self.data_config.root_dir) / era5_rel_path
                            if era5_file.exists():
                                file_pair['era5'] = era5_file
                    
                        file_pairs.append(file_pair)
                    else:
                        missing = []
                        if not cape_file.exists():
                            missing.append(str(cape_file))
                        if not lightning_file.exists():
                            missing.append(str(lightning_file))
                        if not terrain_file.exists():
                            missing.append(str(terrain_file))
                        logger.warning(f"Missing files: {missing}")
    
        return file_pairs
    
    def _setup_preprocessor(self):
        """Setup data preprocessor with normalization statistics."""
    
        # Use default stats or load from file if available
        cape_stats = {'mean': 0.0, 'std': 1.0}
        terrain_stats = {'mean': 0.0, 'std': 1.0}
    
        # Try to load stats from file
        stats_file = Path(self.data_config.root_dir) / 'normalization_stats.yaml'
        if stats_file.exists():
            try:
                import yaml
                with open(stats_file, 'r') as f:
                    stats = yaml.safe_load(f)
                    cape_stats = stats.get('cape', cape_stats)
                    terrain_stats = stats.get('terrain', terrain_stats)
            except Exception as e:
                logger.warning(f"Failed to load stats from {stats_file}: {e}")
    
        self.preprocessor = DataPreprocessor(
            cape_stats=cape_stats,
            terrain_stats=terrain_stats
        )
    
        logger.info("Data preprocessor initialized")
    
    def _setup_augmentations(self):
        """Setup data augmentation pipelines."""
        
        # Spatial augmentation
        if getattr(self.data_config, 'spatial_augmentation', {}).get('enabled', False):
            spatial_config = self.data_config.spatial_augmentation
            self.spatial_augmentation = SpatialAugmentation(
                rotation_range=spatial_config.get('rotation_range', 15),
                translation_range=spatial_config.get('translation_range', 0.1),
                scale_range=spatial_config.get('scale_range', 0.1),
                horizontal_flip=spatial_config.get('horizontal_flip', True),
                vertical_flip=spatial_config.get('vertical_flip', False)
            )
            logger.info("Spatial augmentation enabled")
        
        # Meteorological augmentation
        if getattr(self.data_config, 'meteorological_augmentation', {}).get('enabled', False):
            met_config = self.data_config.meteorological_augmentation
            self.meteorological_augmentation = MeteorologicalAugmentation(
                cape_noise_std=met_config.get('cape_noise_std', 50.0),
                cape_bias_range=met_config.get('cape_bias_range', 100.0),
                temporal_jitter=met_config.get('temporal_jitter', True)
            )
            logger.info("Meteorological augmentation enabled")
    
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
        if self.test_files:
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
   
    @memory_checkpoint("TRAIN_DATALOADER")
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with memory tracking."""
        
        trace_memory_line()  # Before dataloader creation
        
        with MemoryContext("TRAIN_DATALOADER_CREATION"):
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=True,
                num_workers=self.data_config.num_workers,
                pin_memory=self.data_config.pin_memory,
                drop_last=True,
                persistent_workers=True if self.data_config.num_workers > 0 else False
            )
        
        trace_memory_line()  # After dataloader creation
        return dataloader
   
    @memory_checkpoint("VAL_DATALOADER")
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with memory tracking."""
        
        trace_memory_line()  # Before dataloader creation
        
        with MemoryContext("VAL_DATALOADER_CREATION"):
            dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=False,
                num_workers=self.data_config.num_workers,
                pin_memory=self.data_config.pin_memory,
                drop_last=False,
                persistent_workers=True if self.data_config.num_workers > 0 else False
            )
        
        trace_memory_line()  # After dataloader creation
        return dataloader
   
    @memory_checkpoint("TEST_DATALOADER")
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader with memory tracking."""
        
        if not self.test_dataset:
            raise RuntimeError("Test dataset not available")
        
        trace_memory_line()  # Before dataloader creation
        
        with MemoryContext("TEST_DATALOADER_CREATION"):
            dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=False,
                num_workers=self.data_config.num_workers,
                pin_memory=self.data_config.pin_memory,
                drop_last=False,
                persistent_workers=True if self.data_config.num_workers > 0 else False
            )
        
        trace_memory_line()  # After dataloader creation
        return dataloader
   
    @memory_checkpoint("SAMPLE_BATCH")
    def get_sample_batch(self, split: str = 'train') -> Dict:
        """
        Get a sample batch for testing/debugging.
        
        Args:
            split: Which split to sample from ('train', 'val', 'test')
            
        Returns:
            Sample batch dictionary
        """
        trace_memory_line()  # Before sample batch
        
        with MemoryContext("GET_SAMPLE_BATCH"):
            if split == 'train' and self.train_dataset:
                sample = self.train_dataset[0]
            elif split == 'val' and self.val_dataset:
                sample = self.val_dataset[0]
            elif split == 'test' and self.test_dataset:
                sample = self.test_dataset[0]
            else:
                raise ValueError(f"Dataset for split '{split}' not available")
        
        trace_memory_line()  # After sample batch
        return sample


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
        
        for file_pair in self.file_pairs:
            try:
                # Open lightning file to get time dimension
                with xr.open_dataset(file_pair['lightning']) as ds:
                    n_times = len(ds.time)
                
                # Create chunks
                for start_idx in range(0, n_times, self.chunk_size):
                    end_idx = min(start_idx + self.chunk_size, n_times)
                    
                    chunk_info = {
                        'file_pair': file_pair,
                        'time_slice': slice(start_idx, end_idx),
                        'n_timesteps': end_idx - start_idx
                    }
                    
                    self.chunk_index.append(chunk_info)
            
            except Exception as e:
                logger.warning(f"Failed to process file {file_pair['lightning']}: {e}")
        
        logger.info(f"Built chunk index with {len(self.chunk_index)} chunks")
   
    def __len__(self):
        """Return number of chunks."""
        return len(self.chunk_index)
   
    def __iter__(self):
        """Iterate over chunks."""
        
        indices = list(range(len(self.chunk_index)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for idx in indices:
            chunk_info = self.chunk_index[idx]
            
            try:
                # Load chunk data
                chunk_data = self._load_chunk(chunk_info)
                
                # Yield batches from chunk
                yield from self._create_batches(chunk_data)
                
            except Exception as e:
                logger.error(f"Failed to load chunk {idx}: {e}")
                continue
   
    def _load_chunk(self, chunk_info: Dict) -> Dict:
        """Load data for a specific chunk."""
        
        file_pair = chunk_info['file_pair']
        time_slice = chunk_info['time_slice']
        
        # Load data with time slicing
        with xr.open_dataset(file_pair['lightning']) as lightning_ds:
            lightning_data = lightning_ds.isel(time=time_slice).load()
        
        with xr.open_dataset(file_pair['cape']) as cape_ds:
            cape_data = cape_ds.isel(time=time_slice).load()
        
        with xr.open_dataset(file_pair['terrain']) as terrain_ds:
            terrain_data = terrain_ds.load()  # Terrain is static
        
        era5_data = None
        if 'era5' in file_pair:
            with xr.open_dataset(file_pair['era5']) as era5_ds:
                era5_data = era5_ds.isel(time=time_slice).load()
        
        return {
            'lightning': lightning_data,
            'cape': cape_data,
            'terrain': terrain_data,
            'era5': era5_data
        }
   
    def _create_batches(self, chunk_data: Dict):
        """Create batches from chunk data."""
        
        n_timesteps = chunk_data['lightning'].sizes['time']
        
        for start_idx in range(0, n_timesteps, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_timesteps)
            
            batch = {}
            for key, data in chunk_data.items():
                if data is not None and 'time' in data.dims:
                    batch[key] = data.isel(time=slice(start_idx, end_idx))
                elif data is not None:
                    # For static data like terrain, repeat for batch
                    batch[key] = data
            
            # Preprocess batch
            processed_batch = self.preprocessor.process_batch(batch)
            
            yield processed_batch


# Additional utilities for data loading
class DataPrefetcher:
    """
    Prefetches data to GPU to overlap data loading with computation.
    """
    
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
    
    def __iter__(self):
        first_batch = True
        
        for batch in self.dataloader:
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_batch = self._move_to_device(batch)
                
                if not first_batch:
                    torch.cuda.current_stream().wait_stream(self.stream)
                
                yield next_batch if first_batch else self.next_batch
                
                if not first_batch:
                    del self.next_batch
                
                self.next_batch = next_batch
                first_batch = False
            else:
                yield self._move_to_device(batch)
    
    def _move_to_device(self, batch):
        """Move batch to device."""
        moved_batch = {}
        for key, value in batch.items():
            if hasattr(value, 'to'):
                moved_batch[key] = value.to(self.device, non_blocking=True)
            else:
                moved_batch[key] = value
        return moved_batch