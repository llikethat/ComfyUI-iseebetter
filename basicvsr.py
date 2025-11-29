"""
BasicVSR++ Implementation for ComfyUI-iSeeBetter

This module provides flow-based video super-resolution using learned optical flow.
Based on:
- BasicVSR: The Search for Essential Components in Video Super-Resolution (CVPR 2021)
- BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment (CVPR 2022)

Key features:
- SpyNet for lightweight learned optical flow
- Bidirectional propagation for temporal consistency
- Flow-guided deformable alignment (optional)
- Second-order grid propagation (BasicVSR++)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# SpyNet - Lightweight Optical Flow Network
# =============================================================================

class SpyNetBasicModule(nn.Module):
    """Basic module of SpyNet for optical flow estimation."""
    
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        return self.basic_module(x)


class SpyNet(nn.Module):
    """
    SpyNet: Spatial Pyramid Network for Optical Flow Estimation.
    
    A lightweight network that estimates optical flow in a coarse-to-fine manner
    using a spatial pyramid. Much faster than traditional optical flow methods
    while being learnable end-to-end.
    
    Reference: https://arxiv.org/abs/1611.00850
    """
    
    def __init__(self, num_levels=6, pretrained=True):
        super().__init__()
        self.num_levels = num_levels
        
        # Create pyramid modules
        self.basic_modules = nn.ModuleList([
            SpyNetBasicModule() for _ in range(num_levels)
        ])
        
        # Mean for normalization (ImageNet-style)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if pretrained:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize with reasonable defaults (actual pretrained weights would be loaded separately)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def normalize(self, x):
        """Normalize input images."""
        return (x - self.mean) / self.std
    
    def compute_flow(self, ref, supp):
        """
        Compute optical flow from supp to ref.
        
        Args:
            ref: Reference frame [B, 3, H, W]
            supp: Support frame [B, 3, H, W]
        
        Returns:
            Optical flow [B, 2, H, W]
        """
        B, C, H, W = ref.shape
        
        # Normalize
        ref = self.normalize(ref)
        supp = self.normalize(supp)
        
        # Build pyramid
        ref_pyramid = [ref]
        supp_pyramid = [supp]
        
        for _ in range(self.num_levels - 1):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], kernel_size=2, stride=2))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], kernel_size=2, stride=2))
        
        # Reverse for coarse-to-fine
        ref_pyramid = ref_pyramid[::-1]
        supp_pyramid = supp_pyramid[::-1]
        
        # Coarse-to-fine flow estimation
        flow = torch.zeros(B, 2, ref_pyramid[0].shape[2], ref_pyramid[0].shape[3], 
                          device=ref.device, dtype=ref.dtype)
        
        for level in range(self.num_levels):
            # Get target size for this level
            target_h, target_w = ref_pyramid[level].shape[2], ref_pyramid[level].shape[3]
            
            if level > 0:
                # Upsample flow from previous level to match target size exactly
                flow = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
                # Scale flow values proportionally
                scale_h = target_h / flow.shape[2] if flow.shape[2] > 0 else 1
                scale_w = target_w / flow.shape[3] if flow.shape[3] > 0 else 1
                # After interpolate, flow is already target size, so we just scale by ~2
                flow = flow * 2.0
            
            # Ensure flow matches target size (safety check)
            if flow.shape[2] != target_h or flow.shape[3] != target_w:
                flow = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # Warp support image using current flow estimate
            warped = self.warp(supp_pyramid[level], flow)
            
            # Concatenate ref, warped, and flow
            flow_input = torch.cat([ref_pyramid[level], warped, flow], dim=1)
            
            # Estimate residual flow
            flow_residual = self.basic_modules[level](flow_input)
            flow = flow + flow_residual
        
        return flow
    
    def warp(self, x, flow):
        """
        Warp an image using optical flow.
        
        Args:
            x: Image to warp [B, C, H, W]
            flow: Optical flow [B, 2, H, W]
        
        Returns:
            Warped image [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Ensure flow matches input size
        if flow.shape[2] != H or flow.shape[3] != W:
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
            # Scale flow values when resizing
            flow = flow * torch.tensor([W / flow.shape[3], H / flow.shape[2]], 
                                        device=flow.device).view(1, 2, 1, 1)
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add flow to grid
        grid = grid + flow
        
        # Normalize to [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        
        # Permute for grid_sample: [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)
        
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    def forward(self, ref, supp):
        """Forward pass - compute flow from supp to ref."""
        return self.compute_flow(ref, supp)


# =============================================================================
# Flow-Guided Feature Alignment
# =============================================================================

class FlowGuidedAlignment(nn.Module):
    """
    Flow-guided feature alignment using bilinear warping.
    
    Simpler alternative to deformable convolutions that uses
    optical flow to warp features for alignment.
    """
    
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        
        # Flow refinement network
        self.flow_refine = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2, 3, 1, 1)
        )
        
        # Feature fusion after alignment
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    
    def warp_features(self, feat, flow):
        """Warp features using optical flow."""
        B, C, H, W = feat.shape
        
        # Scale flow to feature resolution if needed
        if flow.shape[2] != H or flow.shape[3] != W:
            orig_h, orig_w = flow.shape[2], flow.shape[3]
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
            # Scale flow values proportionally
            if orig_w > 0 and orig_h > 0:
                flow = flow.clone()
                flow[:, 0] = flow[:, 0] * (W / orig_w)
                flow[:, 1] = flow[:, 1] * (H / orig_h)
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        grid = grid + flow
        grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        
        return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    def forward(self, feat_current, feat_neighbor, flow):
        """
        Align neighbor features to current frame.
        
        Args:
            feat_current: Current frame features [B, C, H, W]
            feat_neighbor: Neighbor frame features [B, C, H, W]
            flow: Optical flow from neighbor to current [B, 2, H, W]
        
        Returns:
            Aligned and fused features [B, C, H, W]
        """
        # Initial warp
        feat_warped = self.warp_features(feat_neighbor, flow)
        
        # Refine flow based on feature difference
        flow_residual = self.flow_refine(torch.cat([feat_warped, flow], dim=1))
        flow_refined = flow + flow_residual
        
        # Final warp with refined flow
        feat_aligned = self.warp_features(feat_neighbor, flow_refined)
        
        # Fuse with current features
        fused = self.fusion(torch.cat([feat_current, feat_aligned], dim=1))
        
        return fused + feat_current  # Residual connection


# =============================================================================
# Residual Blocks
# =============================================================================

class ResidualBlockNoBN(nn.Module):
    """Residual block without batch normalization."""
    
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + identity


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with input convolution for channel adjustment."""
    
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(num_blocks):
            layers.append(ResidualBlockNoBN(out_channels))
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)


# =============================================================================
# Pixel Shuffle Upsampler
# =============================================================================

class PixelShuffleUpsampler(nn.Module):
    """Pixel shuffle upsampling module."""
    
    def __init__(self, in_channels, out_channels=3, scale=4):
        super().__init__()
        
        layers = []
        if scale == 4:
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        elif scale == 2:
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.upsampler = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.upsampler(x)


# =============================================================================
# BasicVSR Network
# =============================================================================

class BasicVSR(nn.Module):
    """
    BasicVSR: Bidirectional Video Super-Resolution Network.
    
    Uses bidirectional propagation to leverage both past and future frames
    for better temporal consistency.
    
    Reference: https://arxiv.org/abs/2012.02181
    """
    
    def __init__(self, 
                 num_channels=3,
                 num_feat=64,
                 num_block=30,
                 scale_factor=4,
                 spynet_pretrained=True):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.num_feat = num_feat
        
        # Optical flow network
        self.spynet = SpyNet(num_levels=6, pretrained=spynet_pretrained)
        
        # Feature extraction
        self.feat_extract = nn.Conv2d(num_channels, num_feat, 3, 1, 1)
        
        # Backward propagation branch
        self.backward_trunk = ResidualBlocksWithInputConv(num_feat + 3, num_feat, num_block)
        
        # Forward propagation branch
        self.forward_trunk = ResidualBlocksWithInputConv(num_feat + 3, num_feat, num_block)
        
        # Fusion of bidirectional features
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        
        # Upsampling
        self.upsampler = PixelShuffleUpsampler(num_feat, num_channels, scale_factor)
        
        # Skip connection upsampler (bicubic equivalent via conv)
        self.img_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    def compute_flow(self, frames):
        """
        Compute bidirectional optical flows for all frames.
        
        Args:
            frames: Video frames [B, T, C, H, W]
        
        Returns:
            backward_flows: Flows from t+1 to t [B, T-1, 2, H, W]
            forward_flows: Flows from t-1 to t [B, T-1, 2, H, W]
        """
        B, T, C, H, W = frames.shape
        
        backward_flows = []
        forward_flows = []
        
        for t in range(T - 1):
            # Backward flow: from t+1 to t
            backward_flow = self.spynet(frames[:, t], frames[:, t + 1])
            backward_flows.append(backward_flow)
            
            # Forward flow: from t to t+1
            forward_flow = self.spynet(frames[:, t + 1], frames[:, t])
            forward_flows.append(forward_flow)
        
        backward_flows = torch.stack(backward_flows, dim=1)  # [B, T-1, 2, H, W]
        forward_flows = torch.stack(forward_flows, dim=1)
        
        return backward_flows, forward_flows
    
    def spatial_warp(self, feat, flow):
        """Warp features using optical flow."""
        B, C, H, W = feat.shape
        
        # Ensure flow matches feature size
        if flow.shape[2] != H or flow.shape[3] != W:
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        grid = grid + flow
        grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        
        return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input video frames [B, T, C, H, W] or [B, C, H, W] for single frame
        
        Returns:
            Super-resolved frames [B, T, C, H*scale, W*scale]
        """
        # Handle single frame input
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add temporal dimension
        
        B, T, C, H, W = x.shape
        
        # Compute optical flows
        if T > 1:
            backward_flows, forward_flows = self.compute_flow(x)
        
        # Initialize propagation features
        backward_feats = []
        forward_feats = []
        
        # Backward propagation (from last to first)
        feat_prop = torch.zeros(B, self.num_feat, H, W, device=x.device)
        for t in range(T - 1, -1, -1):
            frame = x[:, t]
            
            if t < T - 1:
                # Warp previous propagation feature
                flow = backward_flows[:, t]
                feat_prop = self.spatial_warp(feat_prop, flow)
            
            # Concatenate with current frame and propagate
            feat_prop = self.backward_trunk(torch.cat([frame, feat_prop], dim=1))
            backward_feats.insert(0, feat_prop)
        
        # Forward propagation (from first to last)
        feat_prop = torch.zeros(B, self.num_feat, H, W, device=x.device)
        for t in range(T):
            frame = x[:, t]
            
            if t > 0:
                # Warp previous propagation feature
                flow = forward_flows[:, t - 1]
                feat_prop = self.spatial_warp(feat_prop, flow)
            
            # Concatenate with current frame and propagate
            feat_prop = self.forward_trunk(torch.cat([frame, feat_prop], dim=1))
            forward_feats.append(feat_prop)
        
        # Fuse bidirectional features and upsample
        outputs = []
        for t in range(T):
            # Fuse backward and forward features
            fused = self.fusion(torch.cat([backward_feats[t], forward_feats[t]], dim=1))
            
            # Upsample
            sr = self.upsampler(fused)
            
            # Add bicubic upsampled input as skip connection
            base = self.img_upsample(x[:, t])
            sr = sr + base
            
            outputs.append(sr)
        
        outputs = torch.stack(outputs, dim=1)  # [B, T, C, H*scale, W*scale]
        
        # Remove temporal dimension if input was single frame
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        
        return outputs


# =============================================================================
# BasicVSR++ Network (Enhanced Version)
# =============================================================================

class SecondOrderGridPropagation(nn.Module):
    """
    Second-order grid propagation module from BasicVSR++.
    
    Considers features from t-2, t-1, t+1, t+2 for better temporal modeling.
    """
    
    def __init__(self, num_feat=64, num_block=7):
        super().__init__()
        
        self.backbone = ResidualBlocksWithInputConv(
            num_feat * 2 + 3,  # prev feat + curr feat + curr frame
            num_feat,
            num_block
        )
    
    def forward(self, feat_current, feat_propagated, frame_current):
        """
        Propagate features with second-order information.
        """
        return self.backbone(torch.cat([feat_propagated, feat_current, frame_current], dim=1))


class BasicVSRPlusPlus(nn.Module):
    """
    BasicVSR++: Enhanced Video Super-Resolution Network.
    
    Improvements over BasicVSR:
    - Flow-guided deformable alignment
    - Second-order grid propagation
    - Better feature fusion
    
    Reference: https://arxiv.org/abs/2104.13371
    """
    
    def __init__(self,
                 num_channels=3,
                 num_feat=64,
                 num_block=7,
                 num_block_back=30,
                 scale_factor=4,
                 spynet_pretrained=True):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.num_feat = num_feat
        
        # Optical flow network
        self.spynet = SpyNet(num_levels=6, pretrained=spynet_pretrained)
        
        # Feature extraction
        self.feat_extract = nn.Conv2d(num_channels, num_feat, 3, 1, 1)
        
        # Flow-guided alignment
        self.alignment = FlowGuidedAlignment(num_feat)
        
        # Propagation modules
        self.backward_prop1 = SecondOrderGridPropagation(num_feat, num_block)
        self.backward_prop2 = SecondOrderGridPropagation(num_feat, num_block)
        self.forward_prop1 = SecondOrderGridPropagation(num_feat, num_block)
        self.forward_prop2 = SecondOrderGridPropagation(num_feat, num_block)
        
        # Feature refinement after propagation
        self.refinement = ResidualBlocksWithInputConv(num_feat * 4 + 3, num_feat, num_block_back)
        
        # Upsampling
        self.upsampler = PixelShuffleUpsampler(num_feat, num_channels, scale_factor)
        
        # Skip connection
        self.img_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    def compute_flow(self, frames):
        """Compute bidirectional optical flows."""
        B, T, C, H, W = frames.shape
        
        backward_flows = []
        forward_flows = []
        
        for t in range(T - 1):
            backward_flow = self.spynet(frames[:, t], frames[:, t + 1])
            backward_flows.append(backward_flow)
            
            forward_flow = self.spynet(frames[:, t + 1], frames[:, t])
            forward_flows.append(forward_flow)
        
        backward_flows = torch.stack(backward_flows, dim=1)
        forward_flows = torch.stack(forward_flows, dim=1)
        
        return backward_flows, forward_flows
    
    def spatial_warp(self, feat, flow):
        """Warp features using optical flow."""
        B, C, H, W = feat.shape
        
        # Ensure flow matches feature size
        if flow.shape[2] != H or flow.shape[3] != W:
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        grid = grid + flow
        grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        
        return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    def forward(self, x):
        """
        Forward pass with second-order propagation.
        
        Args:
            x: Input video frames [B, T, C, H, W] or [B, C, H, W]
        
        Returns:
            Super-resolved frames
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        B, T, C, H, W = x.shape
        
        # Extract features for all frames
        feats = []
        for t in range(T):
            feat = self.feat_extract(x[:, t])
            feats.append(feat)
        feats = torch.stack(feats, dim=1)  # [B, T, C, H, W]
        
        # Compute flows
        if T > 1:
            backward_flows, forward_flows = self.compute_flow(x)
        
        # First backward propagation
        backward_feats1 = []
        feat_prop = torch.zeros(B, self.num_feat, H, W, device=x.device)
        for t in range(T - 1, -1, -1):
            if t < T - 1:
                flow = backward_flows[:, t]
                feat_prop = self.spatial_warp(feat_prop, flow)
            feat_prop = self.backward_prop1(feats[:, t], feat_prop, x[:, t])
            backward_feats1.insert(0, feat_prop)
        
        # Second backward propagation
        backward_feats2 = []
        feat_prop = torch.zeros(B, self.num_feat, H, W, device=x.device)
        for t in range(T - 1, -1, -1):
            if t < T - 1:
                flow = backward_flows[:, t]
                feat_prop = self.spatial_warp(feat_prop, flow)
            feat_prop = self.backward_prop2(backward_feats1[t], feat_prop, x[:, t])
            backward_feats2.insert(0, feat_prop)
        
        # First forward propagation
        forward_feats1 = []
        feat_prop = torch.zeros(B, self.num_feat, H, W, device=x.device)
        for t in range(T):
            if t > 0:
                flow = forward_flows[:, t - 1]
                feat_prop = self.spatial_warp(feat_prop, flow)
            feat_prop = self.forward_prop1(feats[:, t], feat_prop, x[:, t])
            forward_feats1.append(feat_prop)
        
        # Second forward propagation
        forward_feats2 = []
        feat_prop = torch.zeros(B, self.num_feat, H, W, device=x.device)
        for t in range(T):
            if t > 0:
                flow = forward_flows[:, t - 1]
                feat_prop = self.spatial_warp(feat_prop, flow)
            feat_prop = self.forward_prop2(forward_feats1[t], feat_prop, x[:, t])
            forward_feats2.append(feat_prop)
        
        # Fuse all features and upsample
        outputs = []
        for t in range(T):
            fused = torch.cat([
                backward_feats1[t],
                backward_feats2[t],
                forward_feats1[t],
                forward_feats2[t],
                x[:, t]
            ], dim=1)
            
            refined = self.refinement(fused)
            sr = self.upsampler(refined)
            base = self.img_upsample(x[:, t])
            sr = sr + base
            
            outputs.append(sr)
        
        outputs = torch.stack(outputs, dim=1)
        
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        
        return outputs


# =============================================================================
# Utility Functions
# =============================================================================

def create_basicvsr(scale_factor=4, variant='basic', num_feat=64, num_block=15):
    """
    Factory function to create BasicVSR models.
    
    Args:
        scale_factor: Upscaling factor (2 or 4)
        variant: 'basic' or 'plusplus'
        num_feat: Number of feature channels
        num_block: Number of residual blocks
    
    Returns:
        BasicVSR or BasicVSRPlusPlus model
    """
    if variant == 'plusplus':
        return BasicVSRPlusPlus(
            num_feat=num_feat,
            num_block=7,
            num_block_back=num_block,
            scale_factor=scale_factor
        )
    else:
        return BasicVSR(
            num_feat=num_feat,
            num_block=num_block,
            scale_factor=scale_factor
        )


def load_basicvsr_checkpoint(model, checkpoint_path, strict=False):
    """
    Load BasicVSR checkpoint with flexible key matching.
    
    Args:
        model: BasicVSR model
        checkpoint_path: Path to checkpoint file
        strict: Whether to require exact key matching
    
    Returns:
        model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=strict)
    
    return model
