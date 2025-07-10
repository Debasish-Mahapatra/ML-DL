"""
Terrain encoder for processing high-resolution elevation data.
"""

import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

class TerrainEncoder(nn.Module):
    """
    Encoder for terrain data with learnable downsampling from 1km to 3km resolution.
    Preserves lightning-relevant topographic features.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 embedding_dim: int = 64,
                 downsample_factor: float = 3.0,  # 1km -> 3km
                 learnable_downsample: bool = True,
                 preserve_gradients: bool = True):
        """
        Initialize terrain encoder.
        
        Args:
            in_channels: Number of input channels (1 for elevation)
            embedding_dim: Dimension of terrain embeddings
            downsample_factor: Factor for downsampling (1km->3km = 3.0)
            learnable_downsample: Whether to use learnable downsampling
            preserve_gradients: Whether to preserve terrain gradients
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.downsample_factor = downsample_factor
        self.learnable_downsample = learnable_downsample
        self.preserve_gradients = preserve_gradients
        
        # Terrain feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Learnable downsampling
        if learnable_downsample:
            # FIX: Calculate actual input channels after gradient concatenation
            downsample_input_channels = embedding_dim
            if preserve_gradients:
                downsample_input_channels += max(1, embedding_dim // 2)  # FIX: Add gradient feature channels
            
            self.downsample = LearnableDownsample(
                downsample_input_channels,  # FIX: Use actual input channels (96 when gradients enabled)
                embedding_dim,              # FIX: Keep output channels as embedding_dim (64)
                downsample_factor
            )
        else:
            self.downsample = nn.AdaptiveAvgPool2d(None)  # Will be set dynamically
        
        # Gradient computation for terrain features
        if preserve_gradients:
            self.gradient_conv = nn.Conv2d(embedding_dim, max(1, embedding_dim // 2), 3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _compute_terrain_gradients(self, terrain_features: torch.Tensor) -> torch.Tensor:
        """Compute terrain gradients for enhanced features."""
        # Get the actual number of channels from the tensor
        num_channels = terrain_features.shape[1]
    
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                      dtype=terrain_features.dtype, device=terrain_features.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                      dtype=terrain_features.dtype, device=terrain_features.device)
    
        # Reshape for grouped convolution - one filter per channel
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(num_channels, 1, 1, 1)
    
        # Compute gradients using grouped convolution
        grad_x = F.conv2d(terrain_features, sobel_x, padding=1, groups=num_channels)
        grad_y = F.conv2d(terrain_features, sobel_y, padding=1, groups=num_channels)
    
        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
        return grad_magnitude
    
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass through terrain encoder.
        
        Args:
            x: Terrain data (batch_size, 1, height, width) at 1km resolution
            target_size: Target spatial size for output (height, width) at 3km resolution
            
        Returns:
            Encoded terrain features at 3km resolution
        """
        # Extract terrain features
        features = self.feature_extractor(x)
        
        # Compute terrain gradients if enabled
        if self.preserve_gradients:
            # FIX: Store original features before gradient computation to avoid channel mismatch
            original_features = features
            gradients = self._compute_terrain_gradients(original_features)  # FIX: Use original features with correct channel count
            gradient_features = self.gradient_conv(gradients)
            features = torch.cat([original_features, gradient_features], dim=1)  # FIX: Concatenate with original features
        
        # Downsample to target resolution
        if self.learnable_downsample:
            features = self.downsample(features, target_size)
        else:
            features = F.adaptive_avg_pool2d(features, target_size)
        
        return features

class LearnableDownsample(nn.Module):
    """Learnable downsampling module that preserves important features."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 downsample_factor: float):
        super().__init__()
        
        self.downsample_factor = downsample_factor
        
        # Learnable convolution for downsampling
        kernel_size = int(downsample_factor) + 1
        stride = int(downsample_factor)
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # Initialize with average pooling weights
        self._initialize_as_average_pool()
    
    def _initialize_as_average_pool(self):
        """Initialize convolution weights to mimic average pooling."""
        with torch.no_grad():
            weight = self.conv.weight
            kernel_size = weight.shape[-1]
            avg_weight = 1.0 / (kernel_size * kernel_size)
            weight.fill_(avg_weight)
    
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Apply learnable downsampling."""
        # Apply learnable convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # Ensure exact target size with interpolation if needed
        current_size = x.shape[-2:]
        if current_size != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x

class TerrainFeatureExtractor(nn.Module):
    """Extract specific terrain features relevant for lightning prediction."""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        # Feature extractors for different terrain characteristics
        self.elevation_processor = nn.Conv2d(1, embedding_dim // 4, 3, padding=1)
        self.slope_processor = nn.Conv2d(1, embedding_dim // 4, 3, padding=1)
        self.aspect_processor = nn.Conv2d(1, embedding_dim // 4, 3, padding=1)
        self.roughness_processor = nn.Conv2d(1, embedding_dim // 4, 3, padding=1)
        
        self.feature_fusion = nn.Conv2d(embedding_dim, embedding_dim, 1)
        
    def _compute_slope(self, elevation: torch.Tensor) -> torch.Tensor:
        """Compute terrain slope."""
        # Gradient computation for slope
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=elevation.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=elevation.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(elevation, sobel_x, padding=1)
        grad_y = F.conv2d(elevation, sobel_y, padding=1)
        
        slope = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return slope
    
    def _compute_aspect(self, elevation: torch.Tensor) -> torch.Tensor:
        """Compute terrain aspect (direction of slope)."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=elevation.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=elevation.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(elevation, sobel_x, padding=1)
        grad_y = F.conv2d(elevation, sobel_y, padding=1)
        
        aspect = torch.atan2(grad_y, grad_x)
        return aspect
    
    def _compute_roughness(self, elevation: torch.Tensor) -> torch.Tensor:
        """Compute terrain roughness."""
        # Use Laplacian for roughness estimation
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                dtype=torch.float32, device=elevation.device)
        laplacian = laplacian.view(1, 1, 3, 3)
        
        roughness = F.conv2d(elevation, laplacian, padding=1)
        roughness = torch.abs(roughness)
        
        return roughness
    
    def forward(self, elevation: torch.Tensor) -> torch.Tensor:
        """Extract comprehensive terrain features."""
        # Compute terrain derivatives
        slope = self._compute_slope(elevation)
        aspect = self._compute_aspect(elevation)
        roughness = self._compute_roughness(elevation)
        
        # Process each feature type
        elev_feat = self.elevation_processor(elevation)
        slope_feat = self.slope_processor(slope)
        aspect_feat = self.aspect_processor(aspect)
        rough_feat = self.roughness_processor(roughness)
        
        # Concatenate and fuse features
        combined = torch.cat([elev_feat, slope_feat, aspect_feat, rough_feat], dim=1)
        features = self.feature_fusion(combined)
        
        return features
