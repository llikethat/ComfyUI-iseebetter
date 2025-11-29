"""
Optical Flow Utilities for iSeeBetter ComfyUI Node

This module provides optical flow computation methods for temporal alignment
in video super-resolution. It supports multiple backends including:
- OpenCV Farneback
- OpenCV DIS (Dense Inverse Search)
- RAFT (if available)
- Simple frame difference (fallback)
"""

import torch
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def compute_flow_opencv_farneback(frame1, frame2, normalize_for_iseebetter=True):
    """
    Compute optical flow using OpenCV's Farneback method.
    
    Args:
        frame1: First frame as numpy array [H, W, C] in range [0, 255]
        frame2: Second frame as numpy array [H, W, C] in range [0, 255]
        normalize_for_iseebetter: If True, scale flow to match pyflow range
    
    Returns:
        Optical flow as numpy array [H, W, 2]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for Farneback optical flow")
    
    # Convert to grayscale
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray1 = frame1.astype(np.uint8)
        gray2 = frame2.astype(np.uint8)
    
    # Compute flow with optimized parameters
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=5,        # More pyramid levels
        winsize=21,      # Larger window for smoother flow
        iterations=5,    # More iterations
        poly_n=7,        # Larger neighborhood
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    
    if normalize_for_iseebetter:
        h, w = gray1.shape[:2]
        flow = flow / max(h, w) * 100
    
    return flow


def compute_flow_opencv_dis(frame1, frame2, normalize_for_iseebetter=True):
    """
    Compute optical flow using OpenCV's DIS (Dense Inverse Search) method.
    This is faster and often more accurate than Farneback.
    
    Args:
        frame1: First frame as numpy array [H, W, C] in range [0, 255]
        frame2: Second frame as numpy array [H, W, C] in range [0, 255]
        normalize_for_iseebetter: If True, scale flow to match pyflow range
    
    Returns:
        Optical flow as numpy array [H, W, 2]
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for DIS optical flow")
    
    # Convert to grayscale
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray1 = frame1.astype(np.uint8)
        gray2 = frame2.astype(np.uint8)
    
    # Create DIS optical flow with highest quality preset
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    dis.setFinestScale(0)  # Use finest scale for better accuracy
    dis.setGradientDescentIterations(25)  # More iterations for refinement
    dis.setPatchSize(8)  # Smaller patches for finer detail
    dis.setPatchStride(3)  # Smaller stride for denser estimation
    flow = dis.calc(gray1, gray2, None)
    
    if normalize_for_iseebetter:
        # iSeeBetter's original pyflow produces flow normalized differently
        # pyflow typically outputs in fractional pixel units with smaller magnitude
        # We scale to match the expected range - this is critical for artifact-free results
        h, w = gray1.shape[:2]
        # Normalize flow magnitude to be relative to image dimensions
        # This makes flow values more consistent across different resolutions
        flow = flow / max(h, w) * 100  # Scale factor to match pyflow behavior
    
    return flow


def compute_flow_simple(frame1, frame2):
    """
    Simple frame difference as a fallback when optical flow is not available.
    Not a true optical flow but provides some temporal information.
    
    Args:
        frame1: First frame as numpy array [H, W, C] in range [0, 255]
        frame2: Second frame as numpy array [H, W, C] in range [0, 255]
    
    Returns:
        Pseudo-flow as numpy array [H, W, 2] (horizontal and vertical gradients of difference)
    """
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = np.mean(frame1, axis=2)
        gray2 = np.mean(frame2, axis=2)
    else:
        gray1 = frame1
        gray2 = frame2
    
    # Compute difference
    diff = gray2.astype(np.float32) - gray1.astype(np.float32)
    
    # Compute gradients as pseudo-flow
    grad_x = np.gradient(diff, axis=1)
    grad_y = np.gradient(diff, axis=0)
    
    flow = np.stack([grad_x, grad_y], axis=-1)
    
    # Scale down - simple flow should have very small magnitude
    flow = flow * 0.1
    
    return flow


def compute_flow_zero(frame1, frame2):
    """
    Return zero optical flow - useful when motion estimation is causing artifacts.
    The model will still work but without temporal motion compensation.
    
    Args:
        frame1: First frame as numpy array [H, W, C]
        frame2: Second frame as numpy array [H, W, C]
    
    Returns:
        Zero flow as numpy array [H, W, 2]
    """
    h, w = frame1.shape[:2]
    return np.zeros((h, w, 2), dtype=np.float32)


def smooth_flow(flow, kernel_size=5):
    """
    Apply Gaussian smoothing to optical flow to reduce artifacts.
    
    Args:
        flow: Optical flow [H, W, 2]
        kernel_size: Size of Gaussian kernel (odd number)
    
    Returns:
        Smoothed flow [H, W, 2]
    """
    if not OPENCV_AVAILABLE:
        return flow
    
    flow_x = cv2.GaussianBlur(flow[:, :, 0], (kernel_size, kernel_size), 0)
    flow_y = cv2.GaussianBlur(flow[:, :, 1], (kernel_size, kernel_size), 0)
    
    return np.stack([flow_x, flow_y], axis=-1)


def warp_frame(frame, flow):
    """
    Warp a frame according to optical flow.
    
    Args:
        frame: Frame to warp [H, W, C] or [B, C, H, W] tensor
        flow: Optical flow [H, W, 2] or [B, 2, H, W] tensor
    
    Returns:
        Warped frame
    """
    if isinstance(frame, torch.Tensor):
        return warp_frame_torch(frame, flow)
    else:
        return warp_frame_numpy(frame, flow)


def warp_frame_numpy(frame, flow):
    """Warp frame using numpy/OpenCV."""
    if not OPENCV_AVAILABLE:
        # Simple fallback without warping
        return frame
    
    h, w = flow.shape[:2]
    
    # Create coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Add flow to coordinates
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)
    
    # Remap
    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, 
                       borderMode=cv2.BORDER_REPLICATE)
    
    return warped


def warp_frame_torch(frame, flow):
    """
    Warp frame using PyTorch grid_sample.
    
    Args:
        frame: [B, C, H, W] tensor
        flow: [B, 2, H, W] tensor (flow_x, flow_y)
    
    Returns:
        Warped frame [B, C, H, W]
    """
    B, C, H, W = frame.shape
    
    # Create base grid
    y_coords = torch.linspace(-1, 1, H, device=frame.device)
    x_coords = torch.linspace(-1, 1, W, device=frame.device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
    
    # Normalize flow to [-1, 1] range
    flow_normalized = flow.clone()
    flow_normalized[:, 0] = flow[:, 0] / (W / 2)  # x flow
    flow_normalized[:, 1] = flow[:, 1] / (H / 2)  # y flow
    
    # Add flow to grid
    flow_grid = flow_normalized.permute(0, 2, 3, 1)  # [B, H, W, 2]
    sample_grid = grid + flow_grid
    
    # Warp using grid_sample
    warped = torch.nn.functional.grid_sample(
        frame, sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return warped


class OpticalFlowEstimator:
    """
    Optical flow estimator class that selects the best available method.
    """
    
    METHODS = ['dis', 'farneback', 'simple', 'zero']
    
    def __init__(self, method='dis', smooth_flow=True, smooth_kernel=5):
        """
        Initialize optical flow estimator.
        
        Args:
            method: One of 'dis', 'farneback', 'simple', 'zero'
            smooth_flow: Apply Gaussian smoothing to reduce artifacts
            smooth_kernel: Kernel size for smoothing (odd number)
        """
        self.method = method
        self.smooth_flow_enabled = smooth_flow
        self.smooth_kernel = smooth_kernel
        
        if method == 'dis' and not OPENCV_AVAILABLE:
            print("Warning: OpenCV not available, falling back to simple method")
            self.method = 'simple'
        elif method == 'farneback' and not OPENCV_AVAILABLE:
            print("Warning: OpenCV not available, falling back to simple method")
            self.method = 'simple'
    
    def compute(self, frame1, frame2):
        """
        Compute optical flow from frame1 to frame2.
        
        Args:
            frame1: Source frame [H, W, C] numpy array (0-255 range)
            frame2: Target frame [H, W, C] numpy array (0-255 range)
        
        Returns:
            Optical flow [H, W, 2]
        """
        if self.method == 'dis':
            flow = compute_flow_opencv_dis(frame1, frame2)
        elif self.method == 'farneback':
            flow = compute_flow_opencv_farneback(frame1, frame2)
        elif self.method == 'zero':
            flow = compute_flow_zero(frame1, frame2)
        else:
            flow = compute_flow_simple(frame1, frame2)
        
        # Apply smoothing to reduce artifacts
        if self.smooth_flow_enabled and self.method not in ['zero', 'simple']:
            flow = smooth_flow(flow, self.smooth_kernel)
        
        return flow
    
    def compute_batch(self, frames, target_idx):
        """
        Compute optical flow from all frames to a target frame.
        
        Args:
            frames: List of frames [H, W, C] numpy arrays
            target_idx: Index of the target frame
        
        Returns:
            List of optical flows, one for each frame except target
        """
        target = frames[target_idx]
        flows = []
        
        for i, frame in enumerate(frames):
            if i != target_idx:
                flow = self.compute(frame, target)
                flows.append(flow)
        
        return flows


def compute_flow_for_iseebetter(frames, target_idx, method='dis'):
    """
    Compute optical flows in the format expected by iSeeBetter/RBPN.
    
    Args:
        frames: List of numpy arrays [H, W, C] in range [0, 255]
        target_idx: Index of the target/center frame
        method: Optical flow method ('dis', 'farneback', 'simple')
    
    Returns:
        Tuple of (neighbor_frames, flows) as torch tensors
    """
    estimator = OpticalFlowEstimator(method=method)
    
    target = frames[target_idx]
    neighbor_frames = []
    flows = []
    
    for i, frame in enumerate(frames):
        if i != target_idx:
            # Compute flow from neighbor to target
            flow = estimator.compute(frame, target)
            
            neighbor_frames.append(frame)
            flows.append(flow)
    
    return neighbor_frames, flows


def tensor_to_numpy_frames(tensor):
    """
    Convert PyTorch tensor to list of numpy frames.
    
    Args:
        tensor: [B, C, H, W] or [T, C, H, W] tensor in [0, 1] range
    
    Returns:
        List of numpy arrays [H, W, C] in [0, 255] range
    """
    frames = []
    for i in range(tensor.shape[0]):
        frame = tensor[i].cpu().numpy()
        if frame.shape[0] == 3:  # C, H, W
            frame = np.transpose(frame, (1, 2, 0))  # H, W, C
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def numpy_frames_to_tensor(frames, device='cuda'):
    """
    Convert list of numpy frames to PyTorch tensor.
    
    Args:
        frames: List of numpy arrays [H, W, C] in [0, 255] range
        device: Target device
    
    Returns:
        Tensor [T, C, H, W] in [0, 1] range
    """
    tensors = []
    for frame in frames:
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = np.transpose(frame, (2, 0, 1))  # C, H, W
        tensors.append(torch.from_numpy(frame))
    return torch.stack(tensors, dim=0).to(device)


def flow_to_tensor(flow, device='cuda'):
    """
    Convert numpy optical flow to PyTorch tensor.
    
    Args:
        flow: Numpy array [H, W, 2]
        device: Target device
    
    Returns:
        Tensor [1, 2, H, W]
    """
    flow_t = np.transpose(flow, (2, 0, 1))  # 2, H, W
    flow_tensor = torch.from_numpy(flow_t.astype(np.float32)).unsqueeze(0)
    return flow_tensor.to(device)
