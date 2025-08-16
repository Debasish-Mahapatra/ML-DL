"""
Fixed Terrain encoder with progressive downsampling and correct channel dimensions.
"""

import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

class TerrainEncoder(nn.Module):
    """
    Optimized Terrain encoder with progressive downsampling.
    Processes 1km terrain data efficiently while preserving fine details.
    FIXED: Correct channel dimensions to output exactly embedding_dim channels.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 embedding_dim: int = 64,
                 downsample_factor: float = 3.0,  # 1km -> 3km
                 learnable_downsample: bool = True,
                 preserve_gradients: bool = True):
        """
        Initialize optimized terrain encoder.
        
        Args:
            in_channels: Number of input channels (1 for elevation)
            embedding_dim: Dimension of terrain embeddings (EXACT output channels)
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
        
        # FIXED: Calculate channels to ensure exact embedding_dim output
        if preserve_gradients:
            # Reserve channels for gradients
            gradient_channels = embedding_dim // 4  # 16 channels for gradients
            remaining_channels = embedding_dim - gradient_channels  # 48 channels for features
        else:
            gradient_channels = 0
            remaining_channels = embedding_dim
        
        # Split remaining channels between main features and fine details
        fine_detail_channels = 16  # Fixed at 16 for fine details
        main_feature_channels = remaining_channels - fine_detail_channels  # 32 for main features
        
        # OPTIMIZATION: Progressive downsampling with skip connections
        
        # Stage 1: Quick initial downsample (2130 -> 1065, 2x reduction)
        self.stage1_downsample = nn.Conv2d(
            in_channels, 16, 
            kernel_size=5, stride=2, padding=2, bias=False
        )
        self.stage1_norm = nn.BatchNorm2d(16)
        
        # Stage 2: Further downsample (1065 -> 710, adaptive)
        self.stage2_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.stage2_norm = nn.BatchNorm2d(32)
        
        # Stage 3: Main feature extraction (now at 710x710 - much faster!)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(32, 40, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(40, main_feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(main_feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # SKIP CONNECTION: Preserve fine details from original 1km data
        self.fine_detail_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=3, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, fine_detail_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fine_detail_channels),
            nn.ReLU(inplace=True)
        )
        
        # Gradient computation for terrain features (if enabled)
        if preserve_gradients:
            self.gradient_conv = nn.Conv2d(
                main_feature_channels + fine_detail_channels, 
                gradient_channels, 
                3, padding=1
            )
        
        # FIXED: Final projection to ensure exact embedding_dim output
        total_intermediate_channels = main_feature_channels + fine_detail_channels
        if preserve_gradients:
            total_intermediate_channels += gradient_channels
        
        if total_intermediate_channels != embedding_dim:
            self.final_projection = nn.Conv2d(total_intermediate_channels, embedding_dim, 1)
        else:
            self.final_projection = None
        
        self._initialize_weights()
        
        print(f"   ✓ Fixed TerrainEncoder initialized:")
        print(f"     - Output channels: {embedding_dim} (guaranteed)")
        print(f"     - Main features: {main_feature_channels}, Fine details: {fine_detail_channels}")
        if preserve_gradients:
            print(f"     - Gradient features: {gradient_channels}")
        print(f"     - Progressive downsampling: 2130→1065→710")
        print(f"     - Expected speedup: ~5-7x")
    
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
        num_channels = terrain_features.shape[1]
    
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=terrain_features.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=terrain_features.device)
    
        # Reshape for grouped convolution
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
        Fixed forward pass through terrain encoder.
        
        Args:
            x: Terrain data (batch_size, 1, 2130, 2130) at 1km resolution
            target_size: Target spatial size for output (height, width) at 3km resolution
            
        Returns:
            Encoded terrain features at 3km resolution (batch_size, embedding_dim, H, W)
        """
        original_input = x
        
        # FAST PATH: Progressive downsampling
        # Stage 1: 2130 -> 1065 (2x downsample)
        x1 = self.stage1_downsample(x)
        x1 = self.stage1_norm(x1)
        x1 = F.relu(x1, inplace=True)
        
        # Stage 2: 1065 -> target_size (adaptive)
        x2 = self.stage2_conv(x1)
        x2 = self.stage2_norm(x2)
        x2 = F.relu(x2, inplace=True)
        
        # Adaptive resize to exact target size
        if x2.shape[-2:] != target_size:
            x2 = F.adaptive_avg_pool2d(x2, target_size)
        
        # Stage 3: Feature extraction at target resolution (much faster!)
        main_features = self.feature_extractor(x2)
        
        # SKIP CONNECTION: Extract fine details directly from original 1km data
        fine_details = self.fine_detail_extractor(original_input)
        
        # Ensure fine details match target size
        if fine_details.shape[-2:] != target_size:
            fine_details = F.adaptive_avg_pool2d(fine_details, target_size)
        
        # Combine main features with fine details
        combined_features = torch.cat([main_features, fine_details], dim=1)
        
        # Compute terrain gradients if enabled
        if self.preserve_gradients:
            gradients = self._compute_terrain_gradients(combined_features)
            gradient_features = self.gradient_conv(gradients)
            final_features = torch.cat([combined_features, gradient_features], dim=1)
        else:
            final_features = combined_features
        
        # FIXED: Ensure exact embedding_dim output
        if self.final_projection is not None:
            final_features = self.final_projection(final_features)
        
        # Verify output dimensions
        assert final_features.shape[1] == self.embedding_dim, f"Expected {self.embedding_dim} channels, got {final_features.shape[1]}"
        
        return final_features

class LearnableDownsample(nn.Module):
    """Learnable downsampling module (kept for compatibility)."""
    
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