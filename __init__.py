"""
ComfyUI-iSeeBetter: Video Super-Resolution Custom Nodes

Implementation of iSeeBetter (Spatio-Temporal Video Super Resolution using 
Recurrent-Generative Back-Projection Networks) and BasicVSR/BasicVSR++ for ComfyUI.

Based on: 
- iSeeBetter: https://github.com/amanchadha/iSeeBetter
- BasicVSR: https://arxiv.org/abs/2012.02181
- BasicVSR++: https://arxiv.org/abs/2104.13371

Nodes provided:
- ISeeBetterModelLoader: Load iSeeBetter/RBPN model weights
- ISeeBetterUpscale: Upscale video frames with temporal processing (supports SpyNet flow)
- ISeeBetterCleanUpscale: Artifact-free upscaling
- ISeeBetterSimpleUpscale: Simplified upscaling without optical flow
- BasicVSRModelLoader: Load BasicVSR/BasicVSR++ models
- BasicVSRUpscale: Upscale with learned optical flow (best quality)

Enhancement nodes:
- WaterDetailEnhance: Neural network enhancement for water textures
- WaterPostProcess: CPU-based post-processing
- ImageSharpener: Unsharp masking

Architecture modules:
- rbpn.py: Main RBPN network (matches original iSeeBetter)
- basicvsr.py: BasicVSR/BasicVSR++ with SpyNet optical flow
- dbpns.py: DBPN-S single-image SR component
- base_networks.py: Building block layers
- optical_flow.py: Optical flow computation utilities
- srgan_enhance.py: SRGAN and water enhancement modules
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS
)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.1.0"
__author__ = "Based on iSeeBetter by Aman Chadha, John Britto, Mani M. Roja; BasicVSR by Kelvin C.K. Chan et al."
