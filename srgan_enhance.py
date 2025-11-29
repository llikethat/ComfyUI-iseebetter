"""
SRGAN Enhancement Module for iSeeBetter

This module provides:
1. SRGAN Discriminator for GAN-based refinement
2. VGG-based Perceptual Loss for texture quality
3. Water-specific detail enhancement
4. Frequency-based texture refinement

Optimized for water bodies: sea, ocean, waterfall, splash, swell, wake
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


# ==============================================================================
# SRGAN Discriminator (from LeftThomas implementation used by iSeeBetter)
# ==============================================================================

class Discriminator(nn.Module):
    """
    SRGAN Discriminator Network
    
    Architecture follows the original SRGAN paper:
    - Conv layers with increasing channels: 64 -> 128 -> 256 -> 512
    - BatchNorm + LeakyReLU activation
    - Dense layers at the end for classification
    """
    
    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super(Discriminator, self).__init__()
        
        def conv_block(in_ch, out_ch, kernel_size, stride, padding, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        # Initial conv without batch norm
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Progressive downsampling blocks
        self.conv2 = conv_block(64, 64, 3, 2, 1)    # /2
        self.conv3 = conv_block(64, 128, 3, 1, 1)
        self.conv4 = conv_block(128, 128, 3, 2, 1)  # /4
        self.conv5 = conv_block(128, 256, 3, 1, 1)
        self.conv6 = conv_block(256, 256, 3, 2, 1)  # /8
        self.conv7 = conv_block(256, 512, 3, 1, 1)
        self.conv8 = conv_block(512, 512, 3, 2, 1)  # /16
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return self.classifier(x)
    
    @property
    def output_shape(self):
        return (1,)


# ==============================================================================
# VGG Feature Extractor for Perceptual Loss
# ==============================================================================

class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    Extracts features from different layers for multi-scale perceptual comparison.
    """
    
    def __init__(self, layer_ids: List[int] = None, use_input_norm: bool = True):
        super(VGGFeatureExtractor, self).__init__()
        
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True)
        
        # Default: use conv4_4 (layer 35) as in SRGAN
        if layer_ids is None:
            layer_ids = [35]  # VGG19 conv4_4 before activation
        
        self.layer_ids = sorted(layer_ids)
        max_layer = max(layer_ids) + 1
        
        # Extract only needed layers
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        
        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalization
            self.register_buffer(
                'mean', 
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
    
    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_ids:
                features.append(x)
        
        return features if len(features) > 1 else features[0]


# ==============================================================================
# Loss Functions
# ==============================================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Combines MSE loss on feature maps from different VGG layers.
    """
    
    def __init__(self, layer_weights: dict = None):
        super(PerceptualLoss, self).__init__()
        
        if layer_weights is None:
            layer_weights = {35: 1.0}  # Default: only conv4_4
        
        self.layer_ids = list(layer_weights.keys())
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_ids=self.layer_ids)
        self.criterion = nn.MSELoss()
    
    def forward(self, sr_img, hr_img):
        sr_features = self.vgg(sr_img)
        hr_features = self.vgg(hr_img)
        
        if not isinstance(sr_features, list):
            sr_features = [sr_features]
            hr_features = [hr_features]
        
        loss = 0
        for i, layer_id in enumerate(self.layer_ids):
            loss += self.layer_weights[layer_id] * self.criterion(
                sr_features[i], hr_features[i]
            )
        
        return loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
    def __init__(self, loss_type: str = 'bce'):
        super(AdversarialLoss, self).__init__()
        
        if loss_type == 'bce':
            self.criterion = nn.BCELoss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)


class TVLoss(nn.Module):
    """
    Total Variation Loss for spatial smoothness.
    Helps reduce artifacts while preserving edges.
    """
    
    def __init__(self, weight: float = 1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


class GeneratorLoss(nn.Module):
    """
    Combined generator loss for iSeeBetter training.
    
    Combines:
    - MSE loss (reconstruction)
    - Perceptual loss (VGG features)
    - Adversarial loss (GAN)
    - TV loss (smoothness)
    """
    
    def __init__(
        self, 
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.006,
        adversarial_weight: float = 0.001,
        tv_weight: float = 2e-8
    ):
        super(GeneratorLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.tv_weight = tv_weight
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        self.tv_loss = TVLoss()
    
    def forward(self, sr_img, hr_img, discriminator_pred):
        # MSE loss
        mse = self.mse_loss(sr_img, hr_img)
        
        # Perceptual loss
        perceptual = self.perceptual_loss(sr_img, hr_img)
        
        # Adversarial loss (generator wants discriminator to think SR is real)
        adversarial = self.adversarial_loss(discriminator_pred, is_real=True)
        
        # TV loss
        tv = self.tv_loss(sr_img)
        
        total_loss = (
            self.mse_weight * mse +
            self.perceptual_weight * perceptual +
            self.adversarial_weight * adversarial +
            self.tv_weight * tv
        )
        
        return total_loss, {
            'mse': mse.item(),
            'perceptual': perceptual.item(),
            'adversarial': adversarial.item(),
            'tv': tv.item()
        }


# ==============================================================================
# Water-Specific Enhancement Module
# ==============================================================================

class WaterDetailEnhancer(nn.Module):
    """
    Water-specific detail enhancement module.
    
    Designed to enhance:
    - Wave patterns and ripples
    - Light reflections and caustics
    - Foam and spray details
    - Underwater texture
    
    Uses frequency-aware processing to enhance water textures.
    """
    
    def __init__(
        self, 
        channels: int = 64,
        num_res_blocks: int = 4,
        use_frequency: bool = True
    ):
        super(WaterDetailEnhancer, self).__init__()
        
        self.use_frequency = use_frequency
        
        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.PReLU()
        )
        
        # Residual blocks for detail learning
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # Frequency attention module (for water patterns)
        if use_frequency:
            self.freq_attention = FrequencyAttention(channels)
        
        # Multi-scale detail extraction
        self.multi_scale = MultiScaleDetail(channels)
        
        # Output projection
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(channels, 3, 3, 1, 1)
        )
        
        # Learnable blend weight
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        identity = x
        
        # Extract features
        feat = self.head(x)
        
        # Apply residual blocks
        res_feat = feat
        for block in self.res_blocks:
            res_feat = block(res_feat)
        
        # Apply frequency attention for water patterns
        if self.use_frequency:
            res_feat = self.freq_attention(res_feat)
        
        # Multi-scale detail
        res_feat = self.multi_scale(res_feat) + feat
        
        # Generate residual enhancement
        enhancement = self.tail(res_feat)
        
        # Blend with input
        blend = torch.sigmoid(self.blend_weight)
        output = identity + blend * enhancement
        
        return output.clamp(0, 1)


class ResidualBlock(nn.Module):
    """Residual block with PReLU activation."""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    
    def forward(self, x):
        return x + self.conv(x)


class FrequencyAttention(nn.Module):
    """
    Frequency-domain attention for water pattern enhancement.
    
    Water has characteristic frequency patterns:
    - Low-freq: overall wave shapes
    - Mid-freq: ripples and reflections  
    - High-freq: foam, spray, caustics
    """
    
    def __init__(self, channels: int):
        super(FrequencyAttention, self).__init__()
        
        # Learnable frequency weights
        self.low_weight = nn.Parameter(torch.tensor(1.0))
        self.mid_weight = nn.Parameter(torch.tensor(1.2))
        self.high_weight = nn.Parameter(torch.tensor(1.5))
        
        # Frequency band processing
        self.low_conv = nn.Conv2d(channels, channels, 7, 1, 3, groups=channels)
        self.mid_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.high_conv = nn.Conv2d(channels, channels, 1, 1, 0)
        
        # Attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract different frequency components
        low = self.low_conv(x)  # Low frequency (smooth)
        mid = self.mid_conv(x)  # Mid frequency
        high = self.high_conv(x)  # High frequency (details)
        
        # Combine with learnable weights
        freq_combined = (
            self.low_weight * low +
            self.mid_weight * mid +
            self.high_weight * high
        )
        
        # Apply channel attention
        attention = self.attention(freq_combined)
        
        return x + freq_combined * attention


class MultiScaleDetail(nn.Module):
    """Multi-scale detail extraction for various water pattern sizes."""
    
    def __init__(self, channels: int):
        super(MultiScaleDetail, self).__init__()
        
        # Different kernel sizes for different scales
        self.scale1 = nn.Conv2d(channels, channels // 4, 3, 1, 1)
        self.scale2 = nn.Conv2d(channels, channels // 4, 5, 1, 2)
        self.scale3 = nn.Conv2d(channels, channels // 4, 7, 1, 3)
        self.scale4 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 2, dilation=2),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.PReLU()
        )
    
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        multi = torch.cat([s1, s2, s3, s4], dim=1)
        return self.fusion(multi)


# ==============================================================================
# Post-Processing Utilities for Water Enhancement
# ==============================================================================

class WaterPostProcessor:
    """
    Post-processing utilities for enhancing water in upscaled images.
    
    These are optional CPU-based enhancements that can be applied
    after the neural network upscaling.
    """
    
    @staticmethod
    def enhance_highlights(
        image: torch.Tensor, 
        strength: float = 0.3,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Enhance bright highlights (water reflections, foam).
        
        Args:
            image: [B, C, H, W] or [B, H, W, C] tensor
            strength: Enhancement strength (0-1)
            threshold: Brightness threshold for highlights
        """
        # Ensure BCHW format
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
            permuted = True
        else:
            permuted = False
        
        # Calculate luminance
        luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        
        # Create highlight mask
        highlight_mask = (luminance > threshold).float().unsqueeze(1)
        highlight_mask = F.interpolate(
            highlight_mask, 
            size=image.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Enhance highlights
        enhanced = image + strength * highlight_mask * (1 - image)
        enhanced = enhanced.clamp(0, 1)
        
        if permuted:
            enhanced = enhanced.permute(0, 2, 3, 1)
        
        return enhanced
    
    @staticmethod
    def enhance_local_contrast(
        image: torch.Tensor,
        kernel_size: int = 7,
        strength: float = 0.5
    ) -> torch.Tensor:
        """
        Enhance local contrast for water texture details.
        
        Uses unsharp masking to enhance fine details.
        """
        # Ensure BCHW format
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
            permuted = True
        else:
            permuted = False
        
        # Create Gaussian blur
        padding = kernel_size // 2
        blur = F.avg_pool2d(
            F.pad(image, (padding, padding, padding, padding), mode='reflect'),
            kernel_size, 
            stride=1
        )
        
        # Unsharp mask
        detail = image - blur
        enhanced = image + strength * detail
        enhanced = enhanced.clamp(0, 1)
        
        if permuted:
            enhanced = enhanced.permute(0, 2, 3, 1)
        
        return enhanced
    
    @staticmethod
    def enhance_color_saturation(
        image: torch.Tensor,
        saturation: float = 1.2,
        blue_boost: float = 1.1
    ) -> torch.Tensor:
        """
        Enhance color saturation, with optional blue channel boost for water.
        """
        # Ensure BCHW format
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
            permuted = True
        else:
            permuted = False
        
        # Calculate luminance
        luminance = (
            0.299 * image[:, 0:1] + 
            0.587 * image[:, 1:2] + 
            0.114 * image[:, 2:3]
        )
        
        # Enhance saturation
        enhanced = luminance + saturation * (image - luminance)
        
        # Boost blue channel for water
        if blue_boost != 1.0:
            enhanced[:, 2:3] = enhanced[:, 2:3] * blue_boost
        
        enhanced = enhanced.clamp(0, 1)
        
        if permuted:
            enhanced = enhanced.permute(0, 2, 3, 1)
        
        return enhanced


# ==============================================================================
# Utility Functions
# ==============================================================================

def load_discriminator(path: str, device: torch.device) -> Discriminator:
    """Load a pre-trained discriminator model."""
    model = Discriminator()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def create_generator_loss(
    mse_weight: float = 1.0,
    perceptual_weight: float = 0.006,
    adversarial_weight: float = 0.001,
    tv_weight: float = 2e-8
) -> GeneratorLoss:
    """Create a generator loss function with specified weights."""
    return GeneratorLoss(
        mse_weight=mse_weight,
        perceptual_weight=perceptual_weight,
        adversarial_weight=adversarial_weight,
        tv_weight=tv_weight
    )
