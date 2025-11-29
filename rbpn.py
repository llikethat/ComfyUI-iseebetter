"""
RBPN (Recurrent Back-Projection Network) for iSeeBetter

Exact copy of the original iSeeBetter rbpn.py
Based on: https://github.com/amanchadha/iSeeBetter/blob/master/rbpn.py
"""

import torch
import torch.nn as nn
from .base_networks import ConvBlock, DeconvBlock, ResnetBlock
from .dbpns import Net as DBPNS


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        self.nFrames = nFrames

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # DBPNS
        self.DBPN = DBPNS(base_filter, feat, num_stages, scale_factor)

        # Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
            for _ in range(n_resblock)
        ]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
            for _ in range(n_resblock)
        ]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
            for _ in range(n_resblock)
        ]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # Reconstruction
        self.output = ConvBlock((nFrames - 1) * feat, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, neigbor, flow):
        # initial feature extraction
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]), 1)))

        # Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])

            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        # Reconstruction
        out = torch.cat(Ht, 1)
        output = self.output(out)

        return output


# Alias for compatibility
RBPN = Net


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, hr_est, hr_img, idx):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(hr_est), self.loss_network(hr_img))
        # Image Loss
        image_loss = self.mse_loss(hr_est, hr_img)
        # TV Loss
        tv_loss = self.tv_loss(hr_est)

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def load_rbpn_model(model_path, device='cuda', scale_factor=4, nFrames=7):
    """
    Load a pre-trained RBPN model.
    
    Args:
        model_path: Path to .pth weights file
        device: Target device
        scale_factor: Upscaling factor (2, 4, or 8)
        nFrames: Number of input frames
    
    Returns:
        Tuple of (model, actual_scale_factor, actual_nframes)
    """
    # Load state dict first to detect actual parameters
    state_dict = torch.load(model_path, map_location=device)

    # Handle DataParallel wrapper
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict

    # Detect scale factor from kernel sizes
    detected_scale = scale_factor
    for key in state_dict.keys():
        if 'DBPN.up1.up_conv1.deconv.weight' in key:
            shape = state_dict[key].shape
            kernel = shape[2]
            if kernel == 6:
                detected_scale = 2
            elif kernel == 8:
                detected_scale = 4
            elif kernel == 12:
                detected_scale = 8
            print(f"[iSeeBetter] Detected scale factor: {detected_scale} (kernel={kernel})")
            break

    if detected_scale != scale_factor:
        print(f"[iSeeBetter] Warning: Using detected scale_factor={detected_scale} instead of requested {scale_factor}")
        scale_factor = detected_scale

    # Detect nFrames from output layer
    if 'output.conv.weight' in state_dict:
        in_channels = state_dict['output.conv.weight'].shape[1]
        # in_channels = (nFrames - 1) * feat, feat is typically 64
        detected_nframes = (in_channels // 64) + 1
        print(f"[iSeeBetter] Output layer input channels: {in_channels}, detected nFrames: {detected_nframes}")
        if detected_nframes != nFrames:
            print(f"[iSeeBetter] Using detected nFrames={detected_nframes} instead of requested {nFrames}")
            nFrames = detected_nframes
    else:
        print(f"[iSeeBetter] Warning: Could not find output.conv.weight to detect nFrames, using {nFrames}")

    print(f"[iSeeBetter] Creating model with scale_factor={scale_factor}, nFrames={nFrames}")

    model = Net(
        num_channels=3,
        base_filter=256,
        feat=64,
        num_stages=3,
        n_resblock=5,
        nFrames=nFrames,
        scale_factor=scale_factor
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model, scale_factor, nFrames
