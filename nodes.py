"""
ComfyUI Custom Nodes for iSeeBetter Video Super-Resolution

This module provides ComfyUI nodes for:
1. Loading iSeeBetter/RBPN models
2. Processing video frames with temporal super-resolution
3. Batch processing with optical flow
4. BasicVSR/BasicVSR++ for learned optical flow (optional)

Based on: https://github.com/amanchadha/iSeeBetter
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

import folder_paths

from .rbpn import Net as RBPN, load_rbpn_model
from .optical_flow import (
    OpticalFlowEstimator, 
    tensor_to_numpy_frames,
    numpy_frames_to_tensor,
    flow_to_tensor,
    compute_flow_for_iseebetter
)
from .srgan_enhance import (
    Discriminator,
    WaterDetailEnhancer,
    WaterPostProcessor,
    PerceptualLoss,
    VGGFeatureExtractor
)

# Import BasicVSR (optional - provides learned optical flow)
try:
    from .basicvsr import (
        BasicVSR,
        BasicVSRPlusPlus,
        SpyNet,
        create_basicvsr,
        load_basicvsr_checkpoint
    )
    BASICVSR_AVAILABLE = True
except ImportError as e:
    print(f"[iSeeBetter] BasicVSR not available: {e}")
    BASICVSR_AVAILABLE = False


# Register model folder
ISEEBETTER_MODELS_DIR = os.path.join(folder_paths.models_dir, "iseebetter")
if not os.path.exists(ISEEBETTER_MODELS_DIR):
    os.makedirs(ISEEBETTER_MODELS_DIR, exist_ok=True)


def get_model_files():
    """Get list of available iSeeBetter model files."""
    model_files = []
    
    # Check custom iseebetter folder
    if os.path.exists(ISEEBETTER_MODELS_DIR):
        for f in os.listdir(ISEEBETTER_MODELS_DIR):
            if f.endswith('.pth') or f.endswith('.pt'):
                model_files.append(f)
    
    # Check upscale_models folder as fallback
    upscale_dir = os.path.join(folder_paths.models_dir, "upscale_models")
    if os.path.exists(upscale_dir):
        for f in os.listdir(upscale_dir):
            if ('iseebetter' in f.lower() or 'rbpn' in f.lower()) and \
               (f.endswith('.pth') or f.endswith('.pt')):
                model_files.append(f"upscale_models/{f}")
    
    if not model_files:
        model_files = ["none"]
    
    return model_files


class ISeeBetterModelLoader:
    """
    Load an iSeeBetter (RBPN) model for video super-resolution.
    
    This node loads pre-trained iSeeBetter/RBPN weights for use
    with the video upscaling nodes.
    
    Note: The model's nFrames is fixed at training time. If the checkpoint
    was trained with nFrames=7, you must use 7 or the weights won't match.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_model_files(),),
                "scale_factor": ([2, 4, 8], {"default": 4}),
            },
            "optional": {
                "force_num_frames": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 15, 
                    "step": 1,
                    "tooltip": "Force specific nFrames (0 = auto-detect from checkpoint). WARNING: Must match checkpoint!"
                }),
            }
        }
    
    RETURN_TYPES = ("ISEEBETTER_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Load an iSeeBetter/RBPN model for video super-resolution"
    
    def load_model(self, model_name, scale_factor, force_num_frames=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name == "none":
            # Create untrained model (for testing or when no weights available)
            num_frames = force_num_frames if force_num_frames > 0 else 7
            print(f"[iSeeBetter] No model file specified, creating untrained model with nFrames={num_frames}")
            model = RBPN(
                num_channels=3,
                base_filter=256,
                feat=64,
                num_stages=3,
                n_resblock=5,
                nFrames=num_frames,
                scale_factor=scale_factor
            )
            model = model.to(device)
            model.eval()
        else:
            # Load model from file
            if model_name.startswith("upscale_models/"):
                model_path = os.path.join(folder_paths.models_dir, model_name)
            else:
                model_path = os.path.join(ISEEBETTER_MODELS_DIR, model_name)
            
            print(f"[iSeeBetter] Loading model from: {model_path}")
            
            # Use force_num_frames if specified, otherwise auto-detect
            nFrames_to_use = force_num_frames if force_num_frames > 0 else 7
            
            # load_rbpn_model returns (model, actual_scale_factor, actual_nframes)
            model, scale_factor, num_frames = load_rbpn_model(
                model_path=model_path,
                device=device,
                scale_factor=scale_factor,
                nFrames=nFrames_to_use
            )
            
            # If user forced a different nFrames, warn them
            if force_num_frames > 0 and force_num_frames != num_frames:
                print(f"[iSeeBetter] ⚠️  WARNING: You requested nFrames={force_num_frames} but checkpoint needs nFrames={num_frames}")
                print(f"[iSeeBetter] ⚠️  Using checkpoint's nFrames={num_frames} to avoid errors")
        
        print(f"[iSeeBetter] Model ready: scale={scale_factor}x, nFrames={num_frames}")
        
        return ({
            "model": model,
            "scale_factor": scale_factor,
            "num_frames": num_frames,
            "device": device
        },)


class BasicVSRModelLoader:
    """
    Load a BasicVSR or BasicVSR++ model for video super-resolution.
    
    BasicVSR uses learned optical flow (SpyNet) instead of external flow,
    which often produces better results without artifacts.
    
    Options:
    - BasicVSR: Bidirectional propagation with SpyNet flow
    - BasicVSR++: Enhanced version with second-order propagation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        variant_options = ["basicvsr", "basicvsr_plusplus"]
        
        return {
            "required": {
                "variant": (variant_options, {
                    "default": "basicvsr",
                    "tooltip": "BasicVSR++ has better quality but uses more memory"
                }),
                "scale_factor": ([2, 4], {"default": 4}),
                "num_features": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 128,
                    "step": 16,
                    "tooltip": "Number of feature channels (higher = better quality, more VRAM)"
                }),
                "num_blocks": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 30,
                    "step": 5,
                    "tooltip": "Number of residual blocks"
                }),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional path to pre-trained checkpoint"
                }),
            }
        }
    
    RETURN_TYPES = ("BASICVSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Load BasicVSR/BasicVSR++ model with learned optical flow"
    
    def load_model(self, variant, scale_factor, num_features, num_blocks, checkpoint_path=""):
        if not BASICVSR_AVAILABLE:
            raise RuntimeError("BasicVSR module not available. Check installation.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[BasicVSR] Creating {variant} model (scale={scale_factor}x, feat={num_features}, blocks={num_blocks})")
        
        # Create model
        if variant == "basicvsr_plusplus":
            model = BasicVSRPlusPlus(
                num_feat=num_features,
                num_block=7,
                num_block_back=num_blocks,
                scale_factor=scale_factor
            )
        else:
            model = BasicVSR(
                num_feat=num_features,
                num_block=num_blocks,
                scale_factor=scale_factor
            )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[BasicVSR] Loading checkpoint from: {checkpoint_path}")
            model = load_basicvsr_checkpoint(model, checkpoint_path, strict=False)
        
        model = model.to(device)
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[BasicVSR] Model created with {num_params:.2f}M parameters")
        
        return ({
            "model": model,
            "variant": variant,
            "scale_factor": scale_factor,
            "device": device
        },)


class BasicVSRUpscale:
    """
    Upscale video frames using BasicVSR with learned optical flow.
    
    This node uses SpyNet to learn optical flow end-to-end, avoiding
    the artifacts that can occur with external optical flow methods.
    
    Features:
    - Bidirectional temporal propagation
    - Learned optical flow (no external flow needed)
    - Better temporal consistency
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BASICVSR_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "chunk_size": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Process frames in chunks to save memory (0 = process all at once)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Upscale video using BasicVSR with learned optical flow"
    
    def upscale(self, model, images, chunk_size=10):
        if not BASICVSR_AVAILABLE:
            raise RuntimeError("BasicVSR module not available.")
        
        model_data = model
        basicvsr_model = model_data["model"]
        scale_factor = model_data["scale_factor"]
        device = model_data["device"]
        
        B, H, W, C = images.shape
        print(f"[BasicVSR] Processing {B} frames at {W}x{H}, scale: {scale_factor}x")
        
        # Convert to BCHW and add temporal dimension
        images_bchw = images.permute(0, 3, 1, 2).to(device)  # [B, C, H, W]
        
        # Check VRAM
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[BasicVSR] GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_mem:.1f}GB")
        
        upscaled_frames = []
        
        with torch.no_grad():
            if chunk_size <= 0 or B <= chunk_size:
                # Process all frames at once
                print(f"[BasicVSR] Processing all {B} frames at once...")
                
                # Add batch dimension for temporal: [1, T, C, H, W]
                x = images_bchw.unsqueeze(0)
                
                try:
                    output = basicvsr_model(x)  # [1, T, C, H*s, W*s]
                    output = output.squeeze(0)  # [T, C, H*s, W*s]
                    output = output.clamp(0, 1)
                    upscaled_frames.append(output.cpu())
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[BasicVSR] OOM - trying chunked processing...")
                        torch.cuda.empty_cache()
                        chunk_size = max(2, B // 4)
                    else:
                        raise e
            
            if not upscaled_frames:  # Either chunk mode or fallback from OOM
                # Process in chunks with overlap for temporal consistency
                overlap = min(2, chunk_size // 2)
                
                for start in range(0, B, chunk_size - overlap):
                    end = min(start + chunk_size, B)
                    
                    # Get chunk
                    chunk = images_bchw[start:end].unsqueeze(0)  # [1, T_chunk, C, H, W]
                    
                    print(f"[BasicVSR] Processing frames {start}-{end-1}...")
                    
                    try:
                        output = basicvsr_model(chunk)
                        output = output.squeeze(0).clamp(0, 1)
                        
                        # Handle overlap (skip overlapping frames except for first chunk)
                        if start > 0 and len(upscaled_frames) > 0:
                            output = output[overlap:]  # Skip overlap frames
                        
                        upscaled_frames.append(output.cpu())
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"[BasicVSR] OOM on chunk, falling back to bicubic...")
                            torch.cuda.empty_cache()
                            
                            fallback = F.interpolate(
                                images_bchw[start:end],
                                scale_factor=scale_factor,
                                mode='bicubic',
                                align_corners=False
                            ).clamp(0, 1)
                            
                            if start > 0 and len(upscaled_frames) > 0:
                                fallback = fallback[overlap:]
                            
                            upscaled_frames.append(fallback.cpu())
                        else:
                            raise e
                    
                    torch.cuda.empty_cache()
        
        # Concatenate all chunks
        del images_bchw
        torch.cuda.empty_cache()
        
        result = torch.cat(upscaled_frames, dim=0)  # [T, C, H*s, W*s]
        
        # Convert back to BHWC
        result = result.permute(0, 2, 3, 1).clamp(0, 1)
        
        print(f"[BasicVSR] Output shape: {list(result.shape)}")
        
        return (result,)


class ISeeBetterUpscale:
    """
    Upscale video frames using iSeeBetter with temporal processing.
    
    This node takes a batch of video frames and upscales them using
    the iSeeBetter model with optical flow-guided temporal alignment.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        flow_methods = ["dis", "farneback", "simple", "zero"]
        if BASICVSR_AVAILABLE:
            flow_methods.insert(0, "spynet")  # Add SpyNet as first (best) option
        
        return {
            "required": {
                "model": ("ISEEBETTER_MODEL",),
                "images": ("IMAGE",),
                "optical_flow_method": (flow_methods, {
                    "default": "spynet" if BASICVSR_AVAILABLE else "dis",
                    "tooltip": "spynet=learned flow (best), dis/farneback=OpenCV, zero=no motion compensation"
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Tile size for processing large images (0 = auto)"
                }),
                "tile_overlap": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 128,
                    "step": 8,
                    "tooltip": "Overlap between tiles to avoid seams"
                }),
                "smooth_flow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply Gaussian smoothing to optical flow (reduces artifacts)"
                }),
                "flow_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for optical flow magnitude (lower = less motion compensation)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Upscale video frames using iSeeBetter with temporal super-resolution"
    
    def upscale(self, model, images, optical_flow_method, tile_size=0, tile_overlap=32, smooth_flow=True, flow_scale=1.0):
        model_data = model
        rbpn_model = model_data["model"]
        scale_factor = model_data["scale_factor"]
        num_frames = model_data["num_frames"]
        device = model_data["device"]
        
        # Images are in format [B, H, W, C] with values 0-1
        B, H, W, C = images.shape
        
        print(f"[iSeeBetter] Processing {B} frames at {W}x{H}, scale factor: {scale_factor}, nFrames: {num_frames}")
        print(f"[iSeeBetter] Flow method: {optical_flow_method}, smooth: {smooth_flow}, flow_scale: {flow_scale}")
        
        # Check available VRAM
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            free_mem_gb = total_mem - allocated
            print(f"[iSeeBetter] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[iSeeBetter] VRAM: {total_mem:.1f}GB total, {free_mem_gb:.1f}GB free")
            
            # For high-VRAM GPUs (16GB+), we can be more aggressive
            high_vram = total_mem >= 16
        else:
            high_vram = False
        
        # Determine tile size based on VRAM and image size
        pixels = H * W
        auto_tile_size = tile_size
        if tile_size == 0:
            if high_vram:
                # 24GB can handle larger tiles or even full image for moderate sizes
                if pixels <= 500000:  # Up to ~700x700 - try without tiling
                    auto_tile_size = 0
                    print(f"[iSeeBetter] High VRAM detected - processing without tiling")
                else:
                    auto_tile_size = 384  # Larger tiles for high VRAM
                    print(f"[iSeeBetter] Using large tiles (tile_size={auto_tile_size}) for high VRAM GPU")
            else:
                # Conservative for lower VRAM
                if pixels > 200000:
                    auto_tile_size = 128
                elif pixels > 100000:
                    auto_tile_size = 192
        
        tile_size = auto_tile_size
        
        # Convert to [B, C, H, W] format and move to GPU
        # Estimate memory needed
        input_mem_gb = (B * C * H * W * 4) / (1024**3)  # float32
        output_mem_gb = (B * C * H * W * scale_factor**2 * 4) / (1024**3)
        print(f"[iSeeBetter] Estimated memory: input={input_mem_gb:.2f}GB, output={output_mem_gb:.2f}GB")
        
        if input_mem_gb > 2.0:
            print(f"[iSeeBetter] Warning: Large input batch. Consider processing in smaller chunks.")
        
        images_bchw = images.permute(0, 3, 1, 2).to(device)
        
        # Initialize optical flow estimator
        use_spynet = optical_flow_method == "spynet" and BASICVSR_AVAILABLE
        spynet_model = None
        
        if use_spynet:
            print("[iSeeBetter] Using SpyNet for learned optical flow (best quality)")
            spynet_model = SpyNet(num_levels=6, pretrained=True).to(device)
            spynet_model.eval()
        else:
            flow_estimator = OpticalFlowEstimator(method=optical_flow_method, smooth_flow=smooth_flow)
        
        # Number of neighbor frames needed (nFrames - 1)
        num_neighbors = num_frames - 1
        center_offset = num_frames // 2
        upscaled_frames = []
        
        print(f"[iSeeBetter] Using {num_neighbors} neighbor frames, tile_size={tile_size if tile_size > 0 else 'full'}")
        
        for i in range(B):
            # Get target frame
            target_frame = images_bchw[i:i+1]  # [1, C, H, W]
            
            # Prepare neighbor frames and compute optical flow
            neighbor_frames_list = []
            flows_list = []
            
            # Gather exactly num_neighbors neighbor frames
            offsets = list(range(-center_offset, 0)) + list(range(1, center_offset + 1))
            
            for offset in offsets:
                idx = max(0, min(B - 1, i + offset))
                neighbor = images_bchw[idx:idx+1]  # Already on GPU
                
                # Compute optical flow
                if use_spynet:
                    # SpyNet computes flow on GPU directly
                    with torch.no_grad():
                        flow_t = spynet_model(target_frame, neighbor)  # [1, 2, H, W]
                        # Apply flow scale
                        flow_t = flow_t * flow_scale
                else:
                    # CPU-based flow estimation
                    target_np = (target_frame[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    neighbor_np = (neighbor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    flow = flow_estimator.compute(neighbor_np, target_np)
                    
                    # Apply flow scale - lower values reduce motion compensation
                    flow = flow * flow_scale
                    
                    # Convert to tensor
                    flow_t = torch.from_numpy(flow.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
                
                # Debug: Print flow statistics on first frame
                if i == 0 and len(flows_list) == 0:
                    print(f"[iSeeBetter] Flow stats (after scale): min={flow_t.min().item():.2f}, max={flow_t.max().item():.2f}, mean={flow_t.mean().item():.2f}")
                
                neighbor_frames_list.append(neighbor)
                flows_list.append(flow_t)
            
            # Debug: Print input statistics on first frame
            if i == 0:
                print(f"[iSeeBetter] Target frame stats: min={target_frame.min().item():.4f}, max={target_frame.max().item():.4f}")
                print(f"[iSeeBetter] Neighbor count: {len(neighbor_frames_list)}, Flow count: {len(flows_list)}")
            
            # Process with model
            with torch.no_grad():
                try:
                    if tile_size > 0:
                        # Tiled processing
                        output = self._process_tiled_v2(
                            rbpn_model, target_frame,
                            neighbor_frames_list, flows_list,
                            tile_size, tile_overlap, scale_factor, device
                        )
                    else:
                        # Full image processing (for high VRAM GPUs)
                        output = rbpn_model(target_frame, neighbor_frames_list, flows_list)
                    
                    # CRITICAL: Clamp output to [0, 1] range immediately
                    # The model may output values slightly outside this range
                    output = output.clamp(0, 1)
                    
                    # Debug: Check output range on first frame
                    if i == 0:
                        print(f"[iSeeBetter] Output stats (before clamp on future frames):")
                        print(f"  - min={output.min().item():.4f}, max={output.max().item():.4f}")
                        print(f"  - mean={output.mean().item():.4f}, std={output.std().item():.4f}")
                        print(f"  - shape={list(output.shape)}")
                        
                        # Check for potential artifacts (very low or high std indicates issues)
                        if output.std().item() < 0.01:
                            print(f"[iSeeBetter] WARNING: Very low output variance - may indicate model issues!")
                        if output.std().item() > 0.5:
                            print(f"[iSeeBetter] WARNING: Very high output variance - may indicate artifacts!")
                        
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if "out of memory" in error_str or "allocation" in error_str:
                        print(f"[iSeeBetter] OOM on frame {i}, switching to tiled processing...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Try with tiles
                        fallback_tile = 256 if high_vram else 128
                        try:
                            output = self._process_tiled_v2(
                                rbpn_model, target_frame,
                                neighbor_frames_list, flows_list,
                                fallback_tile, 32, scale_factor, device
                            )
                            output = output.clamp(0, 1)
                        except RuntimeError:
                            print(f"[iSeeBetter] Still OOM with tiles, trying smaller...")
                            torch.cuda.empty_cache()
                            try:
                                output = self._process_tiled_v2(
                                    rbpn_model, target_frame,
                                    neighbor_frames_list, flows_list,
                                    128, 16, scale_factor, device
                                )
                                output = output.clamp(0, 1)
                            except:
                                print(f"[iSeeBetter] Falling back to bicubic for frame {i}")
                                output = F.interpolate(
                                    target_frame, scale_factor=scale_factor,
                                    mode='bicubic', align_corners=False
                                ).clamp(0, 1)
                    else:
                        print(f"[iSeeBetter] Error on frame {i}: {e}")
                        output = F.interpolate(
                            target_frame, scale_factor=scale_factor,
                            mode='bicubic', align_corners=False
                        ).clamp(0, 1)
            
            upscaled_frames.append(output.cpu())  # Move to CPU immediately
            
            # Cleanup flow tensors (neighbors are views of images_bchw)
            del flows_list, output
            
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                print(f"[iSeeBetter] Processed {i + 1}/{B} frames")
        
        # Free input images from GPU
        del images_bchw
        torch.cuda.empty_cache()
        
        # Stack all frames (on CPU)
        result = torch.cat(upscaled_frames, dim=0)
        
        # Convert back to [B, H, W, C] format
        result = result.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        
        print(f"[iSeeBetter] Output shape: {result.shape}")
        
        return (result,)
    
    def _process_tiled(self, model, target, neighbors, flows, 
                       tile_size, overlap, scale_factor, device):
        """Process large images in tiles to avoid OOM."""
        _, C, H, W = target.shape
        out_H, out_W = H * scale_factor, W * scale_factor
        
        output = torch.zeros(1, C, out_H, out_W, device=device)
        weight = torch.zeros(1, 1, out_H, out_W, device=device)
        
        # Calculate number of tiles
        step = tile_size - overlap
        n_tiles_h = (H + step - 1) // step
        n_tiles_w = (W + step - 1) // step
        
        for ty in range(n_tiles_h):
            for tx in range(n_tiles_w):
                # Input coordinates
                y1 = ty * step
                x1 = tx * step
                y2 = min(y1 + tile_size, H)
                x2 = min(x1 + tile_size, W)
                
                # Adjust start if we hit the edge
                if y2 - y1 < tile_size and ty > 0:
                    y1 = max(0, y2 - tile_size)
                if x2 - x1 < tile_size and tx > 0:
                    x1 = max(0, x2 - tile_size)
                
                # Extract tiles
                target_tile = target[:, :, y1:y2, x1:x2]
                neighbor_tiles = [n[:, :, y1:y2, x1:x2] for n in 
                                  [neighbors[i:i+1] for i in range(neighbors.shape[0])]]
                flow_tiles = [f[:, :, y1:y2, x1:x2] for f in 
                              [flows[i:i+1] for i in range(flows.shape[0])]]
                
                # Process tile
                try:
                    tile_out = model(target_tile, neighbor_tiles, flow_tiles)
                except:
                    tile_out = F.interpolate(
                        target_tile, scale_factor=scale_factor, 
                        mode='bicubic', align_corners=False
                    )
                
                # Output coordinates
                out_y1, out_y2 = y1 * scale_factor, y2 * scale_factor
                out_x1, out_x2 = x1 * scale_factor, x2 * scale_factor
                
                # Create weight mask for blending
                tile_h, tile_w = tile_out.shape[2:]
                w_mask = torch.ones(1, 1, tile_h, tile_w, device=device)
                
                # Add to output with weighting
                output[:, :, out_y1:out_y2, out_x1:out_x2] += tile_out * w_mask
                weight[:, :, out_y1:out_y2, out_x1:out_x2] += w_mask
        
        # Normalize by weights
        output = output / (weight + 1e-8)
        
        return output

    def _process_tiled_v2(self, model, target, neighbors_list, flows_list,
                          tile_size, overlap, scale_factor, device):
        """
        Memory-efficient tiled processing with improved blending.
        Works with data on GPU or CPU.
        """
        _, C, H, W = target.shape
        out_H, out_W = H * scale_factor, W * scale_factor
        
        # Output buffer on GPU
        output = torch.zeros(1, C, out_H, out_W, device=device)
        weight = torch.zeros(1, 1, out_H, out_W, device=device)
        
        # Calculate number of tiles - ensure overlap is reasonable
        effective_overlap = min(overlap, tile_size // 4)  # Don't let overlap be too large
        step = max(1, tile_size - effective_overlap)
        n_tiles_h = max(1, (H + step - 1) // step)
        n_tiles_w = max(1, (W + step - 1) // step)
        
        total_tiles = n_tiles_h * n_tiles_w
        
        for ty in range(n_tiles_h):
            for tx in range(n_tiles_w):
                # Input coordinates
                y1 = ty * step
                x1 = tx * step
                y2 = min(y1 + tile_size, H)
                x2 = min(x1 + tile_size, W)
                
                # Adjust start if we hit the edge
                if y2 - y1 < tile_size and ty > 0:
                    y1 = max(0, y2 - tile_size)
                if x2 - x1 < tile_size and tx > 0:
                    x1 = max(0, x2 - tile_size)
                
                # Extract target tile
                target_tile = target[:, :, y1:y2, x1:x2]
                
                # Extract neighbor and flow tiles, ensure on GPU
                neighbor_tiles = []
                flow_tiles = []
                
                for n, f in zip(neighbors_list, flows_list):
                    n_tile = n[:, :, y1:y2, x1:x2]
                    f_tile = f[:, :, y1:y2, x1:x2]
                    # Move to device if not already there
                    if n_tile.device != device:
                        n_tile = n_tile.to(device)
                    if f_tile.device != device:
                        f_tile = f_tile.to(device)
                    neighbor_tiles.append(n_tile)
                    flow_tiles.append(f_tile)
                
                # Process tile
                try:
                    tile_out = model(target_tile, neighbor_tiles, flow_tiles)
                    # Clamp tile output immediately
                    tile_out = tile_out.clamp(0, 1)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        # Fall back to bicubic for this tile
                        tile_out = F.interpolate(
                            target_tile, scale_factor=scale_factor,
                            mode='bicubic', align_corners=False
                        ).clamp(0, 1)
                    else:
                        raise e
                
                # Output coordinates
                out_y1, out_y2 = y1 * scale_factor, y2 * scale_factor
                out_x1, out_x2 = x1 * scale_factor, x2 * scale_factor
                
                # Create weight mask for blending (feathered edges)
                tile_h, tile_w = tile_out.shape[2:]
                w_mask = self._create_blend_mask(tile_h, tile_w, effective_overlap * scale_factor, device)
                
                # Add to output with weighting
                output[:, :, out_y1:out_y2, out_x1:out_x2] += tile_out * w_mask
                weight[:, :, out_y1:out_y2, out_x1:out_x2] += w_mask
                
                # Clean up tile tensors
                del tile_out, neighbor_tiles, flow_tiles
        
        # Normalize by weights and clamp final output
        output = output / (weight + 1e-8)
        output = output.clamp(0, 1)
        
        return output
    
    def _create_blend_mask(self, h, w, fade_size, device):
        """Create a feathered blend mask for seamless tile blending."""
        mask = torch.ones(1, 1, h, w, device=device)
        
        # Use a reasonable fade size - not too large
        fade_size = min(fade_size, min(h, w) // 4)
        
        if fade_size > 2:
            # Create smooth fade using cosine interpolation for better quality
            fade = torch.linspace(0, 1, fade_size, device=device)
            # Use cosine fade for smoother transition
            fade = (1 - torch.cos(fade * 3.14159)) / 2
            
            # Apply to edges
            mask[:, :, :fade_size, :] *= fade.view(1, 1, -1, 1)
            mask[:, :, -fade_size:, :] *= fade.flip(0).view(1, 1, -1, 1)
            mask[:, :, :, :fade_size] *= fade.view(1, 1, 1, -1)
            mask[:, :, :, -fade_size:] *= fade.flip(0).view(1, 1, 1, -1)
        
        return mask


class ISeeBetterSimpleUpscale:
    """
    Simplified iSeeBetter upscaling without explicit optical flow.
    
    This node uses the RBPN model but with zero optical flow,
    still leveraging multi-frame information for better results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale_factor": ([2, 4], {"default": 4}),
                "num_frames": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 9,
                    "step": 2,
                    "tooltip": "Number of frames to use for temporal context"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Simplified iSeeBetter upscaling with zero optical flow"
    
    def upscale(self, images, scale_factor, num_frames):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B, H, W, C = images.shape
        print(f"[iSeeBetter Simple] Processing {B} frames at {W}x{H}")
        
        # Create RBPN model
        model = RBPN(
            num_channels=3,
            base_filter=256,
            feat=64,
            num_stages=3,
            n_resblock=5,
            nFrames=num_frames,
            scale_factor=scale_factor
        ).to(device)
        model.eval()
        
        # Convert to BCHW
        images_bchw = images.permute(0, 3, 1, 2).to(device)
        
        center_offset = num_frames // 2
        upscaled_frames = []
        
        with torch.no_grad():
            for i in range(B):
                # Get target frame
                target = images_bchw[i:i+1]
                
                # Gather neighbor frames
                neighbors = []
                flows = []
                
                for j in range(-center_offset, center_offset + 1):
                    if j != 0:  # Skip target
                        idx = max(0, min(B - 1, i + j))
                        neighbors.append(images_bchw[idx:idx+1])
                        # Zero optical flow
                        flows.append(torch.zeros(1, 2, H, W, device=device))
                
                # Run model
                try:
                    output = model(target, neighbors, flows)
                except Exception as e:
                    print(f"[iSeeBetter Simple] Model error: {e}")
                    print("[iSeeBetter Simple] Falling back to bicubic")
                    output = torch.nn.functional.interpolate(
                        target, scale_factor=scale_factor,
                        mode='bicubic', align_corners=False
                    )
                
                upscaled_frames.append(output)
                
                if (i + 1) % 20 == 0:
                    print(f"[iSeeBetter Simple] Processed {i + 1}/{B} frames")
        
        result = torch.cat(upscaled_frames, dim=0)
        result = result.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        
        return (result,)


class ISeeBetterFrameBuffer:
    """
    Buffer node to collect frames for iSeeBetter processing.
    
    Use this to batch frames from a video for processing with
    the temporal upscaling nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "buffer_size": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 31,
                    "step": 2,
                    "tooltip": "Number of frames to buffer"
                }),
                "mode": (["sliding_window", "batch"], {
                    "default": "sliding_window",
                    "tooltip": "sliding_window: process overlapping windows, batch: process all at once"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("buffered_frames", "center_index")
    FUNCTION = "buffer_frames"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Buffer video frames for iSeeBetter temporal processing"
    
    def buffer_frames(self, image, buffer_size, mode):
        B = image.shape[0]
        center_idx = buffer_size // 2
        
        if mode == "batch":
            # Just pass through all frames
            return (image, center_idx)
        else:
            # For sliding window, we'd typically handle this differently
            # in a more complex workflow. Here we just return the input.
            return (image, center_idx)


class ISeeBetterCleanUpscale:
    """
    Artifact-free iSeeBetter upscaling with optimized processing.
    
    This node is designed to minimize visual artifacts like "golf ball" patterns
    that can occur with standard processing. It uses:
    - Optimized optical flow that matches iSeeBetter's training data
    - No tiling (processes full frame for consistent results)
    - Batch processing to reduce per-frame overhead
    
    Best for: Videos where standard upscaling produces artifacts
    Limitations: May require more VRAM than tiled processing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ISEEBETTER_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "use_temporal_info": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use neighbor frames for temporal consistency. Disable if seeing artifacts"
                }),
                "blend_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend between model output (1.0) and bicubic fallback (0.0)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Artifact-free iSeeBetter upscaling without tiling"
    
    def upscale(self, model, images, use_temporal_info=True, blend_factor=1.0):
        model_data = model
        rbpn_model = model_data["model"]
        scale_factor = model_data["scale_factor"]
        num_frames = model_data["num_frames"]
        device = model_data["device"]
        
        B, H, W, C = images.shape
        print(f"[iSeeBetter Clean] Processing {B} frames at {W}x{H}, scale: {scale_factor}x")
        print(f"[iSeeBetter Clean] Temporal info: {use_temporal_info}, blend: {blend_factor}")
        
        # Convert to BCHW
        images_bchw = images.permute(0, 3, 1, 2).to(device)
        
        center_offset = num_frames // 2
        num_neighbors = num_frames - 1
        upscaled_frames = []
        
        with torch.no_grad():
            for i in range(B):
                target = images_bchw[i:i+1]
                
                # Prepare neighbors and flows
                neighbors = []
                flows = []
                
                offsets = list(range(-center_offset, 0)) + list(range(1, center_offset + 1))
                
                for offset in offsets:
                    idx = max(0, min(B - 1, i + offset))
                    neighbor = images_bchw[idx:idx+1]
                    neighbors.append(neighbor)
                    
                    if use_temporal_info and idx != i:
                        # Simple motion estimation - compute pixel-wise difference
                        # This is more stable than optical flow for some content
                        diff = (neighbor - target)
                        # Convert difference to pseudo-flow (gradient of difference)
                        grad_x = torch.nn.functional.conv2d(
                            diff.mean(dim=1, keepdim=True),
                            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=diff.dtype, device=device) / 8.0,
                            padding=1
                        )
                        grad_y = torch.nn.functional.conv2d(
                            diff.mean(dim=1, keepdim=True),
                            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=diff.dtype, device=device) / 8.0,
                            padding=1
                        )
                        flow = torch.cat([grad_x, grad_y], dim=1)
                        # Scale to very small values - this prevents artifacts
                        flow = flow * 0.1
                    else:
                        # Zero flow when not using temporal info or same frame
                        flow = torch.zeros(1, 2, H, W, device=device)
                    
                    flows.append(flow)
                
                # Process through model
                try:
                    output = rbpn_model(target, neighbors, flows)
                    output = output.clamp(0, 1)
                    
                    # Optionally blend with bicubic upscale for smoother results
                    if blend_factor < 1.0:
                        bicubic = F.interpolate(
                            target, scale_factor=scale_factor,
                            mode='bicubic', align_corners=False
                        ).clamp(0, 1)
                        output = bicubic * (1 - blend_factor) + output * blend_factor
                    
                except RuntimeError as e:
                    print(f"[iSeeBetter Clean] Error on frame {i}: {e}")
                    print("[iSeeBetter Clean] Falling back to bicubic")
                    output = F.interpolate(
                        target, scale_factor=scale_factor,
                        mode='bicubic', align_corners=False
                    ).clamp(0, 1)
                
                upscaled_frames.append(output.cpu())
                
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    print(f"[iSeeBetter Clean] Processed {i + 1}/{B} frames")
        
        # Stack results
        del images_bchw
        torch.cuda.empty_cache()
        
        result = torch.cat(upscaled_frames, dim=0)
        result = result.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        
        print(f"[iSeeBetter Clean] Output shape: {list(result.shape)}")
        
        return (result,)


class WaterDetailEnhance:
    """
    Enhance water-specific details in upscaled images.
    
    Uses neural network-based enhancement optimized for:
    - Wave patterns and ripples
    - Light reflections and caustics
    - Foam and spray details
    - Underwater textures
    
    Best used after iSeeBetter upscaling for additional detail recovery.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Enhancement strength (0 = no effect, 1 = full effect)"
                }),
                "use_frequency_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use frequency-aware processing for water patterns"
                }),
            },
            "optional": {
                "enhancer_model": ("WATER_ENHANCER_MODEL",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_images",)
    FUNCTION = "enhance"
    CATEGORY = "video/upscaling/water"
    DESCRIPTION = "Enhance water-specific details (waves, reflections, foam)"
    
    def enhance(self, images, strength, use_frequency_attention, enhancer_model=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B, H, W, C = images.shape
        print(f"[Water Enhance] Processing {B} images at {W}x{H}")
        
        # Use provided model or create new one
        if enhancer_model is not None:
            model = enhancer_model["model"]
        else:
            model = WaterDetailEnhancer(
                channels=64,
                num_res_blocks=4,
                use_frequency=use_frequency_attention
            ).to(device)
            model.eval()
        
        # Convert to BCHW
        images_bchw = images.permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            enhanced = model(images_bchw)
        
        # Blend with original based on strength
        if strength < 1.0:
            enhanced = images_bchw * (1 - strength) + enhanced * strength
        
        # Convert back to BHWC
        result = enhanced.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        
        return (result,)


class WaterPostProcess:
    """
    Post-processing for water images.
    
    Apply various enhancements optimized for water bodies:
    - Highlight enhancement for reflections
    - Local contrast for texture details
    - Color saturation with optional blue boost
    
    This is a fast CPU-based processing step.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "highlight_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Enhance bright reflections and foam"
                }),
                "contrast_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Enhance local texture contrast"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Color saturation multiplier"
                }),
                "blue_boost": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.8,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Blue channel boost for water color"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process"
    CATEGORY = "video/upscaling/water"
    DESCRIPTION = "Post-processing enhancements for water imagery"
    
    def process(self, images, highlight_strength, contrast_strength, 
                saturation, blue_boost):
        
        result = images.clone()
        
        # Apply enhancements in order
        if highlight_strength > 0:
            result = WaterPostProcessor.enhance_highlights(
                result, strength=highlight_strength
            )
        
        if contrast_strength > 0:
            result = WaterPostProcessor.enhance_local_contrast(
                result, strength=contrast_strength
            )
        
        if saturation != 1.0 or blue_boost != 1.0:
            result = WaterPostProcessor.enhance_color_saturation(
                result, saturation=saturation, blue_boost=blue_boost
            )
        
        return (result.clamp(0, 1),)


class WaterEnhancerModelLoader:
    """
    Load a pre-trained water detail enhancer model.
    
    This allows loading custom-trained models for specific
    water enhancement tasks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_water_enhancer_models(),),
            },
            "optional": {
                "channels": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 128,
                    "step": 16,
                    "tooltip": "Number of feature channels"
                }),
                "num_res_blocks": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of residual blocks"
                }),
            }
        }
    
    RETURN_TYPES = ("WATER_ENHANCER_MODEL",)
    RETURN_NAMES = ("enhancer_model",)
    FUNCTION = "load_model"
    CATEGORY = "video/upscaling/water"
    DESCRIPTION = "Load a water detail enhancer model"
    
    def load_model(self, model_name, channels=64, num_res_blocks=4):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = WaterDetailEnhancer(
            channels=channels,
            num_res_blocks=num_res_blocks,
            use_frequency=True
        ).to(device)
        
        if model_name != "none":
            # Load pre-trained weights if available
            model_path = os.path.join(ISEEBETTER_MODELS_DIR, model_name)
            if os.path.exists(model_path):
                print(f"[Water Enhancer] Loading model from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
        
        model.eval()
        
        return ({
            "model": model,
            "device": device,
            "channels": channels
        },)


class SRGANDiscriminatorLoader:
    """
    Load an SRGAN discriminator model.
    
    The discriminator can be used for GAN-based refinement
    or quality assessment of upscaled images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_discriminator_models(),),
            }
        }
    
    RETURN_TYPES = ("SRGAN_DISCRIMINATOR",)
    RETURN_NAMES = ("discriminator",)
    FUNCTION = "load_model"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Load an SRGAN discriminator model"
    
    def load_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = Discriminator().to(device)
        
        if model_name != "none":
            model_path = os.path.join(ISEEBETTER_MODELS_DIR, model_name)
            if os.path.exists(model_path):
                print(f"[SRGAN] Loading discriminator from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
        
        model.eval()
        
        return ({
            "model": model,
            "device": device
        },)


class PerceptualQualityScore:
    """
    Calculate perceptual quality score between images.
    
    Uses VGG features to compute perceptual similarity,
    which better reflects human perception than PSNR/SSIM.
    
    Lower score = more similar images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("perceptual_loss", "report")
    FUNCTION = "compute_score"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Calculate perceptual similarity between image sets"
    
    def compute_score(self, images_a, images_b):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check shapes match
        if images_a.shape != images_b.shape:
            return (float('inf'), "Error: Image shapes don't match")
        
        # Convert to BCHW
        a = images_a.permute(0, 3, 1, 2).to(device)
        b = images_b.permute(0, 3, 1, 2).to(device)
        
        # Create perceptual loss calculator
        perceptual_loss = PerceptualLoss().to(device)
        
        with torch.no_grad():
            loss = perceptual_loss(a, b)
        
        loss_value = loss.item()
        
        # Generate report
        report = f"Perceptual Loss: {loss_value:.6f}\n"
        report += f"Images compared: {images_a.shape[0]}\n"
        report += f"Resolution: {images_a.shape[2]}x{images_a.shape[1]}\n"
        
        if loss_value < 0.01:
            report += "Quality: Excellent (very similar)"
        elif loss_value < 0.05:
            report += "Quality: Good"
        elif loss_value < 0.1:
            report += "Quality: Moderate"
        else:
            report += "Quality: Poor (significant differences)"
        
        return (loss_value, report)


class ImageSharpener:
    """
    Apply unsharp masking to sharpen images.
    
    Good for enhancing fine details in water textures
    after upscaling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength"
                }),
                "radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Blur radius for unsharp mask"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Only sharpen pixels with difference above threshold"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sharpened_images",)
    FUNCTION = "sharpen"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Apply unsharp masking to sharpen images"
    
    def sharpen(self, images, amount, radius, threshold):
        # Convert to BCHW
        images_bchw = images.permute(0, 3, 1, 2)
        
        # Create Gaussian kernel
        kernel_size = radius * 2 + 1
        sigma = radius / 2.0
        
        # Simple box blur approximation
        padding = radius
        blurred = F.avg_pool2d(
            F.pad(images_bchw, (padding, padding, padding, padding), mode='reflect'),
            kernel_size,
            stride=1
        )
        
        # Calculate difference
        diff = images_bchw - blurred
        
        # Apply threshold
        if threshold > 0:
            mask = (diff.abs() > threshold).float()
            diff = diff * mask
        
        # Apply sharpening
        sharpened = images_bchw + amount * diff
        
        # Convert back to BHWC
        result = sharpened.permute(0, 2, 3, 1).clamp(0, 1)
        
        return (result,)


class GolfBallArtifactRemover:
    """
    Remove "golf ball" / blocky artifacts from video frames.
    
    This node applies various deblocking and smoothing techniques to
    reduce compression artifacts and upscaling artifacts that create
    a bumpy "golf ball" texture.
    
    Techniques used:
    - Bilateral filtering (edge-preserving smoothing)
    - Adaptive deblocking
    - Local contrast normalization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Artifact removal strength (higher = more smoothing)"
                }),
                "preserve_edges": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Edge preservation (higher = keep more detail)"
                }),
                "method": (["bilateral", "guided", "median", "adaptive"], {
                    "default": "bilateral",
                    "tooltip": "bilateral=best quality, guided=fast, median=aggressive, adaptive=auto"
                }),
            },
            "optional": {
                "block_size": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 32,
                    "step": 4,
                    "tooltip": "Block size to detect (8 for most compression artifacts)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cleaned_images",)
    FUNCTION = "remove_artifacts"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Remove golf ball / blocky artifacts from video frames"
    
    def remove_artifacts(self, images, strength, preserve_edges, method, block_size=8):
        """Apply artifact removal."""
        B, H, W, C = images.shape
        print(f"[ArtifactRemover] Processing {B} frames at {W}x{H}, method={method}")
        
        # Convert to BCHW
        images_bchw = images.permute(0, 3, 1, 2)
        
        if method == "bilateral":
            result = self._bilateral_filter(images_bchw, strength, preserve_edges)
        elif method == "guided":
            result = self._guided_filter(images_bchw, strength, preserve_edges)
        elif method == "median":
            result = self._median_filter(images_bchw, strength)
        elif method == "adaptive":
            result = self._adaptive_deblock(images_bchw, strength, preserve_edges, block_size)
        else:
            result = images_bchw
        
        # Blend with original based on strength
        result = images_bchw * (1 - strength) + result * strength
        
        # Convert back to BHWC
        result = result.permute(0, 2, 3, 1).clamp(0, 1)
        
        return (result,)
    
    def _bilateral_filter(self, x, strength, edge_preserve):
        """Edge-preserving bilateral-like filter using convolutions."""
        B, C, H, W = x.shape
        
        # Create distance-based spatial kernel
        kernel_size = int(5 + strength * 6)  # 5 to 11
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma_spatial = kernel_size / 3
        sigma_range = (1 - edge_preserve) * 0.3 + 0.05  # Range sigma
        
        # For simplicity, use a separable approximation
        # First pass: spatial smoothing
        padding = kernel_size // 2
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - padding
        gaussian_1d = torch.exp(-coords**2 / (2 * sigma_spatial**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # Apply separable Gaussian
        kernel_h = gaussian_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        kernel_v = gaussian_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        
        # Horizontal pass
        x_pad = F.pad(x, (padding, padding, 0, 0), mode='reflect')
        x_smooth = F.conv2d(x_pad, kernel_h, groups=C)
        
        # Vertical pass
        x_pad = F.pad(x_smooth, (0, 0, padding, padding), mode='reflect')
        x_smooth = F.conv2d(x_pad, kernel_v, groups=C)
        
        # Adaptive blending based on local variance (edge detection)
        local_var = self._local_variance(x, kernel_size)
        edge_mask = torch.sigmoid((local_var - 0.01) * 100 * edge_preserve)
        
        # Blend: keep original at edges, use smoothed in flat areas
        result = x * edge_mask + x_smooth * (1 - edge_mask)
        
        return result
    
    def _guided_filter(self, x, strength, edge_preserve):
        """Guided filter for edge-preserving smoothing."""
        B, C, H, W = x.shape
        
        radius = int(2 + strength * 8)  # 2 to 10
        eps = (1 - edge_preserve) * 0.1 + 0.001
        
        # Use box filter approximation
        kernel_size = 2 * radius + 1
        
        # Mean filter
        mean_I = F.avg_pool2d(
            F.pad(x, (radius, radius, radius, radius), mode='reflect'),
            kernel_size, stride=1
        )
        
        mean_II = F.avg_pool2d(
            F.pad(x * x, (radius, radius, radius, radius), mode='reflect'),
            kernel_size, stride=1
        )
        
        var_I = mean_II - mean_I * mean_I
        
        # Guided filter coefficients
        a = var_I / (var_I + eps)
        b = mean_I - a * mean_I
        
        # Mean of coefficients
        mean_a = F.avg_pool2d(
            F.pad(a, (radius, radius, radius, radius), mode='reflect'),
            kernel_size, stride=1
        )
        mean_b = F.avg_pool2d(
            F.pad(b, (radius, radius, radius, radius), mode='reflect'),
            kernel_size, stride=1
        )
        
        result = mean_a * x + mean_b
        
        return result
    
    def _median_filter(self, x, strength):
        """Median filter for aggressive artifact removal."""
        B, C, H, W = x.shape
        
        kernel_size = int(3 + strength * 4)  # 3 to 7
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        padding = kernel_size // 2
        
        # Unfold to get patches
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        patches = x_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        # patches: [B, C, H, W, k, k]
        
        patches = patches.contiguous().view(B, C, H, W, -1)
        
        # Take median
        result = patches.median(dim=-1)[0]
        
        return result
    
    def _adaptive_deblock(self, x, strength, edge_preserve, block_size):
        """Adaptive deblocking based on block boundary detection."""
        B, C, H, W = x.shape
        
        # Detect block boundaries
        block_mask = self._detect_block_boundaries(x, block_size)
        
        # Apply stronger smoothing at block boundaries
        smooth = self._bilateral_filter(x, strength * 1.5, edge_preserve)
        
        # Blend based on block boundary detection
        result = x * (1 - block_mask * strength) + smooth * (block_mask * strength)
        
        return result
    
    def _detect_block_boundaries(self, x, block_size):
        """Detect blocking artifact boundaries."""
        B, C, H, W = x.shape
        
        # Compute gradient magnitude
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Pad back to original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        # Create block boundary mask
        mask_x = torch.zeros_like(x[:, :1, :, :])
        mask_y = torch.zeros_like(x[:, :1, :, :])
        
        for i in range(block_size, W, block_size):
            if i < W:
                mask_x[:, :, :, i] = 1
        for i in range(block_size, H, block_size):
            if i < H:
                mask_y[:, :, i, :] = 1
        
        # Combine with gradient to find actual block artifacts
        block_mask = (grad_x.mean(dim=1, keepdim=True) * mask_x + 
                      grad_y.mean(dim=1, keepdim=True) * mask_y)
        
        # Dilate the mask slightly
        block_mask = F.max_pool2d(block_mask, 3, stride=1, padding=1)
        
        # Normalize
        block_mask = block_mask / (block_mask.max() + 1e-8)
        
        return block_mask.expand(-1, C, -1, -1)
    
    def _local_variance(self, x, kernel_size):
        """Compute local variance for edge detection."""
        padding = kernel_size // 2
        
        # Mean
        mean = F.avg_pool2d(
            F.pad(x, (padding, padding, padding, padding), mode='reflect'),
            kernel_size, stride=1
        )
        
        # Mean of squares
        mean_sq = F.avg_pool2d(
            F.pad(x * x, (padding, padding, padding, padding), mode='reflect'),
            kernel_size, stride=1
        )
        
        # Variance = E[X^2] - E[X]^2
        var = mean_sq - mean * mean
        
        return var


# Helper functions for model discovery
def get_water_enhancer_models():
    """Get list of available water enhancer model files."""
    model_files = ["none"]
    
    if os.path.exists(ISEEBETTER_MODELS_DIR):
        for f in os.listdir(ISEEBETTER_MODELS_DIR):
            if 'water' in f.lower() and (f.endswith('.pth') or f.endswith('.pt')):
                model_files.append(f)
    
    return model_files


def get_discriminator_models():
    """Get list of available discriminator model files."""
    model_files = ["none"]
    
    if os.path.exists(ISEEBETTER_MODELS_DIR):
        for f in os.listdir(ISEEBETTER_MODELS_DIR):
            if 'discriminator' in f.lower() or 'netD' in f.lower():
                if f.endswith('.pth') or f.endswith('.pt'):
                    model_files.append(f)
    
    return model_files


class ISeeBetterDebugTest:
    """
    Debug node for testing iSeeBetter model with minimal processing.
    
    This node processes a single image with zero optical flow to help
    isolate whether artifacts are from the model or from processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ISEEBETTER_MODEL",),
                "image": ("IMAGE",),
                "use_zero_flow": ("BOOLEAN", {"default": True, "tooltip": "Use zero optical flow instead of computing it"}),
                "force_no_tiling": ("BOOLEAN", {"default": True, "tooltip": "Force processing without tiling"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "test_upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Debug test for iSeeBetter - processes single frame to isolate issues"
    
    def test_upscale(self, model, image, use_zero_flow, force_no_tiling):
        model_data = model
        rbpn_model = model_data["model"]
        scale_factor = model_data["scale_factor"]
        num_frames = model_data["num_frames"]
        device = model_data["device"]
        
        # Take just first frame if batch
        if len(image.shape) == 4:
            single_image = image[0:1]  # [1, H, W, C]
        else:
            single_image = image.unsqueeze(0)
        
        B, H, W, C = single_image.shape
        print(f"\n[iSeeBetter Debug] Testing single frame {W}x{H}")
        print(f"[iSeeBetter Debug] scale_factor={scale_factor}, nFrames={num_frames}")
        print(f"[iSeeBetter Debug] Input: min={single_image.min():.4f}, max={single_image.max():.4f}")
        
        # Convert to BCHW
        target = single_image.permute(0, 3, 1, 2).to(device)  # [1, C, H, W]
        
        # Create neighbor frames (duplicates of target)
        num_neighbors = num_frames - 1
        neighbor_frames = [target.clone() for _ in range(num_neighbors)]
        
        # Create flows
        if use_zero_flow:
            print(f"[iSeeBetter Debug] Using ZERO optical flow")
            flows = [torch.zeros(1, 2, H, W, device=device) for _ in range(num_neighbors)]
        else:
            print(f"[iSeeBetter Debug] Computing actual optical flow (all same frame)")
            # Even with same frame, compute flow to test flow computation
            target_np = (target[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            from .optical_flow import OpticalFlowEstimator
            flow_estimator = OpticalFlowEstimator(method='dis')
            flows = []
            for _ in range(num_neighbors):
                flow = flow_estimator.compute(target_np, target_np)
                flow_t = torch.from_numpy(flow.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
                flows.append(flow_t)
                print(f"[iSeeBetter Debug] Flow stats: min={flow.min():.4f}, max={flow.max():.4f}")
        
        # Process with model
        print(f"[iSeeBetter Debug] Running model inference...")
        with torch.no_grad():
            try:
                output = rbpn_model(target, neighbor_frames, flows)
                
                print(f"[iSeeBetter Debug] Raw output: min={output.min():.4f}, max={output.max():.4f}")
                print(f"[iSeeBetter Debug] Raw output: mean={output.mean():.4f}, std={output.std():.4f}")
                
                # Clamp output
                output = output.clamp(0, 1)
                
                print(f"[iSeeBetter Debug] Clamped output: min={output.min():.4f}, max={output.max():.4f}")
                print(f"[iSeeBetter Debug] Output shape: {list(output.shape)}")
                
            except RuntimeError as e:
                print(f"[iSeeBetter Debug] ERROR: {e}")
                print(f"[iSeeBetter Debug] Falling back to bicubic...")
                output = F.interpolate(target, scale_factor=scale_factor, mode='bicubic', align_corners=False)
                output = output.clamp(0, 1)
        
        # Convert back to BHWC
        result = output.permute(0, 2, 3, 1).cpu()
        
        print(f"[iSeeBetter Debug] Final output: {list(result.shape)}")
        print(f"[iSeeBetter Debug] Test complete!\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ISeeBetterModelLoader": ISeeBetterModelLoader,
    "ISeeBetterUpscale": ISeeBetterUpscale,
    "ISeeBetterCleanUpscale": ISeeBetterCleanUpscale,
    "ISeeBetterSimpleUpscale": ISeeBetterSimpleUpscale,
    "ISeeBetterFrameBuffer": ISeeBetterFrameBuffer,
    "WaterDetailEnhance": WaterDetailEnhance,
    "WaterPostProcess": WaterPostProcess,
    "WaterEnhancerModelLoader": WaterEnhancerModelLoader,
    "SRGANDiscriminatorLoader": SRGANDiscriminatorLoader,
    "PerceptualQualityScore": PerceptualQualityScore,
    "ImageSharpener": ImageSharpener,
    "GolfBallArtifactRemover": GolfBallArtifactRemover,
    "ISeeBetterDebugTest": ISeeBetterDebugTest,
}

# Add BasicVSR nodes if available
if BASICVSR_AVAILABLE:
    NODE_CLASS_MAPPINGS["BasicVSRModelLoader"] = BasicVSRModelLoader
    NODE_CLASS_MAPPINGS["BasicVSRUpscale"] = BasicVSRUpscale

NODE_DISPLAY_NAME_MAPPINGS = {
    "ISeeBetterModelLoader": "iSeeBetter Model Loader",
    "ISeeBetterUpscale": "iSeeBetter Video Upscale",
    "ISeeBetterCleanUpscale": "iSeeBetter Clean Upscale (No Artifacts)",
    "ISeeBetterSimpleUpscale": "iSeeBetter Simple Upscale",
    "ISeeBetterFrameBuffer": "iSeeBetter Frame Buffer",
    "WaterDetailEnhance": "Water Detail Enhance",
    "WaterPostProcess": "Water Post Process",
    "WaterEnhancerModelLoader": "Water Enhancer Model Loader",
    "SRGANDiscriminatorLoader": "SRGAN Discriminator Loader",
    "PerceptualQualityScore": "Perceptual Quality Score",
    "ImageSharpener": "Image Sharpener",
    "GolfBallArtifactRemover": "Golf Ball Artifact Remover",
    "ISeeBetterDebugTest": "iSeeBetter Debug Test",
}

# Add BasicVSR display names if available
if BASICVSR_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS["BasicVSRModelLoader"] = "BasicVSR Model Loader"
    NODE_DISPLAY_NAME_MAPPINGS["BasicVSRUpscale"] = "BasicVSR Video Upscale (Learned Flow)"
