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

# MEMORY TRACKING IMPORTS - EXISTING
from ..utils.memory_tracker import memory_checkpoint, trace_memory_line, MemoryContext

# DEBUG UTILITIES IMPORTS - NEW ADDITION
from ..utils.debug_utils import get_debug_manager, debug_print, is_debug_enabled

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
        
        # Initialize debug manager - NEW
        self.debug_manager = get_debug_manager(config)
        
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
        if self.debug_manager.verbose_logging:
            debug_print("Preparing data...", "verbose")
        else:
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
                if self.debug_manager.verbose_logging:
                    debug_print(f"Split file not found: {split_path}", "verbose")
                else:
                    logger.warning(f"Split file not found: {split_path}")
                # Could auto-generate splits here if needed
    
    @memory_checkpoint("DATAMODULE_SETUP")
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        This is called on every process.
        """
        if self.debug_manager.verbose_logging:
            debug_print(f"Setting up data for stage: {stage}", "verbose")
        else:
            logger.info(f"Setting up data for stage: {stage}")
        
        # Conditional memory tracing - MODIFIED
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("START_OF_SETUP")
        
        # Load file splits
        with MemoryContext("LOAD_FILE_SPLITS"):
            self._load_file_splits()
        
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("AFTER_FILE_SPLITS_LOADING")
        
        # Initialize preprocessor with normalization stats
        with MemoryContext("SETUP_PREPROCESSOR"):
            self._setup_preprocessor()
        
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("AFTER_PREPROCESSOR_SETUP")
        
        # Initialize augmentations
        with MemoryContext("SETUP_AUGMENTATIONS"):
            self._setup_augmentations()
        
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("AFTER_AUGMENTATION_SETUP")
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            with MemoryContext("SETUP_TRAIN_VAL_DATASETS"):
                self._setup_train_val_datasets()
            
            if self.debug_manager.memory_tracking:
                self.debug_manager.conditional_trace_memory("AFTER_TRAIN_VAL_DATASET_SETUP")
        
        if stage == 'test' or stage is None:
            with MemoryContext("SETUP_TEST_DATASET"):
                self._setup_test_dataset()
            
            if self.debug_manager.memory_tracking:
                self.debug_manager.conditional_trace_memory("AFTER_TEST_DATASET_SETUP")
    
    def _load_file_splits(self):
        """Load file pairs for each split."""
        
        splits_dir = Path(self.data_config.splits_dir)
        
        # Load training files
        train_file = splits_dir / 'train_files.txt'
        if train_file.exists():
            self.train_files = self._parse_file_list(train_file)
            
            if self.debug_manager.verbose_logging:
                debug_print(f"Loaded {len(self.train_files)} training file pairs", "verbose")
            else:
                logger.info(f"Loaded {len(self.train_files)} training file pairs")
        else:
            raise FileNotFoundError(f"Training split file not found: {train_file}")
        
        # Load validation files
        val_file = splits_dir / 'val_files.txt'
        if val_file.exists():
            self.val_files = self._parse_file_list(val_file)
            
            if self.debug_manager.verbose_logging:
                debug_print(f"Loaded {len(self.val_files)} validation file pairs", "verbose")
            else:
                logger.info(f"Loaded {len(self.val_files)} validation file pairs")
        else:
            raise FileNotFoundError(f"Validation split file not found: {val_file}")
        
        # Load test files (optional)
        test_file = splits_dir / 'test_files.txt'
        if test_file.exists():
            self.test_files = self._parse_file_list(test_file)
            
            if self.debug_manager.verbose_logging:
                debug_print(f"Loaded {len(self.test_files)} test file pairs", "verbose")
            else:
                logger.info(f"Loaded {len(self.test_files)} test file pairs")
        else:
            if self.debug_manager.verbose_logging:
                debug_print("Test split file not found - test evaluation will be skipped", "verbose")
            else:
                logger.warning("Test split file not found - test evaluation will be skipped")
            self.test_files = []
    
    def _parse_file_list(self, file_path: Path) -> List[Dict[str, Path]]:
        """Parse file list from text file - CORRECTED FOR OPTION A."""
        file_pairs = []
        
        # Define the single terrain file path (same for all samples)
        terrain_file = Path(self.data_config.root_dir) / 'terrain/terrain_odisha_1km.nc'
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse line format: cape_file,lightning_file (2-column format)
                    parts = line.split(',')
                    if len(parts) == 2:
                        file_pairs.append({
                            'cape': Path(self.data_config.root_dir) / parts[0].strip(),
                            'lightning': Path(self.data_config.root_dir) / parts[1].strip(),
                            'terrain': terrain_file  # Same terrain file for all samples
                        })
                    else:
                        if self.debug_manager.verbose_logging:
                            debug_print(f"Skipping invalid line in {file_path}: {line}", "verbose")
                        else:
                            logger.warning(f"Skipping invalid line in {file_path}: {line}")
        
        return file_pairs
    
    def _setup_preprocessor(self):
        """Initialize data preprocessor with normalization statistics - CORRECTED."""
        
        if self.debug_manager.verbose_logging:
            debug_print("Setting up data preprocessor", "verbose")
        
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
                    
                if self.debug_manager.verbose_logging:
                    debug_print(f"Loaded normalization stats from {stats_file}", "verbose")
            except Exception as e:
                if self.debug_manager.verbose_logging:
                    debug_print(f"Failed to load stats from {stats_file}: {e}", "verbose")
                else:
                    logger.warning(f"Failed to load stats from {stats_file}: {e}")

        # CORRECTED: Use the actual constructor parameters
        self.preprocessor = DataPreprocessor(
            cape_stats=cape_stats,
            terrain_stats=terrain_stats
        )
        
        if self.debug_manager.verbose_logging:
            debug_print("Data preprocessor setup complete", "verbose")
    
    def _setup_augmentations(self):
        """Initialize augmentation transforms."""
        
        if not getattr(self.data_config, 'augmentation', {}).get('enabled', False):
            if self.debug_manager.verbose_logging:
                debug_print("Data augmentation disabled", "verbose")
            return
        
        if self.debug_manager.verbose_logging:
            debug_print("Setting up data augmentations", "verbose")
        
        # Spatial augmentation
        if getattr(self.data_config.augmentation, 'spatial', {}).get('enabled', False):
            spatial_config = self.data_config.augmentation.spatial
            self.spatial_augmentation = SpatialAugmentation(
                rotation_range=getattr(spatial_config, 'rotation_range', 15),
                scale_range=getattr(spatial_config, 'scale_range', (0.9, 1.1)),
                translation_range=getattr(spatial_config, 'translation_range', 0.1),
                flip_probability=getattr(spatial_config, 'flip_probability', 0.5)
            )
            
            if self.debug_manager.verbose_logging:
                debug_print("Spatial augmentation enabled", "verbose")
        
        # Meteorological augmentation
        if getattr(self.data_config.augmentation, 'meteorological', {}).get('enabled', False):
            met_config = self.data_config.augmentation.meteorological
            self.meteorological_augmentation = MeteorologicalAugmentation(
                noise_std=getattr(met_config, 'noise_std', 0.1),
                cape_scaling_range=getattr(met_config, 'cape_scaling_range', (0.95, 1.05)),
                dropout_probability=getattr(met_config, 'dropout_probability', 0.1)
            )
            
            if self.debug_manager.verbose_logging:
                debug_print("Meteorological augmentation enabled", "verbose")
    
    def _setup_train_val_datasets(self):
        """Setup training and validation datasets."""
        
        if self.debug_manager.verbose_logging:
            debug_print("Setting up training and validation datasets", "verbose")
        
        # Get target shapes from config with fallbacks
        target_cape_shape = tuple(getattr(self.data_config.domain, 'grid_size_25km', (85, 85)))
        target_lightning_shape = tuple(getattr(self.data_config.domain, 'grid_size_3km', (710, 710)))
        target_terrain_shape = tuple(getattr(self.data_config.domain, 'grid_size_1km', (2130, 2130)))
        
        # Training dataset (with augmentation)
        self.train_dataset = LightningDataset(
            file_pairs=self.train_files,
            preprocessor=self.preprocessor,
            sequence_length=getattr(self.data_config.temporal, 'sequence_length', 1),
            spatial_augmentation=self.spatial_augmentation,
            meteorological_augmentation=self.meteorological_augmentation,
            target_cape_shape=target_cape_shape,
            target_lightning_shape=target_lightning_shape,
            target_terrain_shape=target_terrain_shape
        )
        
        if self.debug_manager.batch_info:
            debug_print(f"Training dataset created with {len(self.train_dataset)} samples", "batch")
        
        # Validation dataset (no augmentation)
        self.val_dataset = LightningDataset(
            file_pairs=self.val_files,
            preprocessor=self.preprocessor,
            sequence_length=getattr(self.data_config.temporal, 'sequence_length', 1),
            spatial_augmentation=None,
            meteorological_augmentation=None,
            target_cape_shape=target_cape_shape,
            target_lightning_shape=target_lightning_shape,
            target_terrain_shape=target_terrain_shape
        )
        
        if self.debug_manager.batch_info:
            debug_print(f"Validation dataset created with {len(self.val_dataset)} samples", "batch")
   
    def _setup_test_dataset(self):
        """Setup test dataset."""
        if self.test_files:
            if self.debug_manager.verbose_logging:
                debug_print("Setting up test dataset", "verbose")
            
            # Get target shapes from config with fallbacks
            target_cape_shape = tuple(getattr(self.data_config.domain, 'grid_size_25km', (85, 85)))
            target_lightning_shape = tuple(getattr(self.data_config.domain, 'grid_size_3km', (710, 710)))
            target_terrain_shape = tuple(getattr(self.data_config.domain, 'grid_size_1km', (2130, 2130)))
            
            self.test_dataset = LightningDataset(
                file_pairs=self.test_files,
                preprocessor=self.preprocessor,
                sequence_length=getattr(self.data_config.temporal, 'sequence_length', 1),
                spatial_augmentation=None,
                meteorological_augmentation=None,
                target_cape_shape=target_cape_shape,
                target_lightning_shape=target_lightning_shape,
                target_terrain_shape=target_terrain_shape
            )
            
            if self.debug_manager.batch_info:
                debug_print(f"Test dataset created with {len(self.test_dataset)} samples", "batch")
   
    @memory_checkpoint("TRAIN_DATALOADER")
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with conditional memory tracking."""
        
        # Conditional memory tracing - MODIFIED
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("BEFORE_TRAIN_DATALOADER_CREATION")
        
        with MemoryContext("TRAIN_DATALOADER_CREATION"):
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=True,
                num_workers=getattr(self.data_config, 'num_workers', 0),
                pin_memory=getattr(self.data_config, 'pin_memory', False),
                drop_last=True,
                persistent_workers=True if getattr(self.data_config, 'num_workers', 0) > 0 else False
            )
        
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("AFTER_TRAIN_DATALOADER_CREATION")
        
        if self.debug_manager.batch_info:
            debug_print(f"Training dataloader created: batch_size={self.data_config.batch_size}, num_workers={getattr(self.data_config, 'num_workers', 0)}", "batch")
        
        return dataloader
   
    @memory_checkpoint("VAL_DATALOADER")
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with conditional memory tracking."""
        
        # Conditional memory tracing - MODIFIED
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("BEFORE_VAL_DATALOADER_CREATION")
        
        with MemoryContext("VAL_DATALOADER_CREATION"):
            dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=False,
                num_workers=getattr(self.data_config, 'num_workers', 0),
                pin_memory=getattr(self.data_config, 'pin_memory', False),
                drop_last=False,
                persistent_workers=True if getattr(self.data_config, 'num_workers', 0) > 0 else False
            )
        
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("AFTER_VAL_DATALOADER_CREATION")
        
        if self.debug_manager.batch_info:
            debug_print(f"Validation dataloader created: batch_size={self.data_config.batch_size}", "batch")
        
        return dataloader
   
    @memory_checkpoint("TEST_DATALOADER")
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader with conditional memory tracking."""
        if self.test_dataset is None:
            if self.debug_manager.verbose_logging:
                debug_print("No test dataset available", "verbose")
            return None
        
        # Conditional memory tracing - MODIFIED
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("BEFORE_TEST_DATALOADER_CREATION")
        
        with MemoryContext("TEST_DATALOADER_CREATION"):
            dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=False,
                num_workers=getattr(self.data_config, 'num_workers', 0),
                pin_memory=getattr(self.data_config, 'pin_memory', False),
                drop_last=False,
                persistent_workers=True if getattr(self.data_config, 'num_workers', 0) > 0 else False
            )
        
        if self.debug_manager.memory_tracking:
            self.debug_manager.conditional_trace_memory("AFTER_TEST_DATALOADER_CREATION")
        
        if self.debug_manager.batch_info:
            debug_print(f"Test dataloader created: batch_size={self.data_config.batch_size}", "batch")
        
        return dataloader
    
    def get_sample_data(self) -> Dict[str, any]:
        """Get a sample data point for model validation."""
        if self.train_dataset is None:
            raise RuntimeError("Datasets not setup. Call setup() first.")
        
        sample = self.train_dataset[0]
        
        if self.debug_manager.batch_info:
            debug_print("Sample data shapes:", "batch")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    debug_print(f"  {key}: {value.shape}", "batch")
                else:
                    debug_print(f"  {key}: {type(value)}", "batch")
        
        return sample
    
    def get_data_statistics(self) -> Dict[str, any]:
        """Get dataset statistics."""
        stats = {}
        
        if self.train_dataset is not None:
            stats['train_size'] = len(self.train_dataset)
        if self.val_dataset is not None:
            stats['val_size'] = len(self.val_dataset)
        if self.test_dataset is not None:
            stats['test_size'] = len(self.test_dataset)
        
        if self.preprocessor is not None:
            stats['normalization_stats'] = {
                'cape_stats': self.preprocessor.cape_stats,
                'terrain_stats': self.preprocessor.terrain_stats
            }
        
        if self.debug_manager.verbose_logging:
            debug_print(f"Dataset statistics: {stats}", "verbose")
        
        return stats