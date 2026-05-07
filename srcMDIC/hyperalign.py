

import torch
import torch.nn as nn
from torch import Tensor
from vector_quantize_pytorch import VectorQuantize
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from compressai.layers import AttentionBlock, conv3x3
from compressai.models.utils import conv, deconv
from einops import rearrange
from einops.layers.torch import Rearrange

torch.set_num_threads(4)
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBottleneckBlock(nn.Module):

    def __init__(self, in_ch: int):
        super().__init__()

        self.layers = nn.Sequential(
            conv1x1(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            conv3x3(in_ch // 2, in_ch // 2),
            nn.ReLU(inplace=True),
            conv1x1(in_ch // 2, in_ch),
        )


    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)

    def forward(self, image: Tensor) -> Tensor:
        return image + self.layers(image)


@dataclass
class HyperAlignOutput(BaseOutput):
    z: torch.FloatTensor = None
    z_hat: torch.FloatTensor = None
    indices: torch.IntTensor = None
    commit_loss: torch.FloatTensor = None

class HyperAlign(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config 
    def __init__(
            self,
            in_channels: int = 4,
            N=192,
            M=320,
            codebook_dim=32,
            cfg_ss=64,
            cfg_cs=256,
    ):
        super().__init__()

        vq_spatialdim = cfg_ss
        self.backbone = nn.Sequential(
            conv1x1(in_channels, N) if vq_spatialdim >= 16 else conv(in_channels, N),
            # conv1x1(in_channels, N),
            # conv(in_channels, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, N) if vq_spatialdim >= 32 else conv(N, N),
            # conv1x1(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv1x1(N, N) if vq_spatialdim == 64 else conv(N, N),
            # conv1x1(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, M),
            AttentionBlock(M)
        )

        # VQ
        self.quantizer = VectorQuantize(dim=M,
                                        codebook_size=cfg_cs,
                                        use_cosine_sim = True,
                                        codebook_dim=codebook_dim)

    def forward(self, z: Tensor) -> Tensor:
        if torch.isnan(z).any():
            print("NaN detected in input!")
        if torch.isinf(z).any():
            print("Inf detected in input!")
        z = self.backbone(z)

        # (B,C,H,W) -> (B,H,W,C)
        z_perm = z.permute(0, 2, 3, 1)
        b, h, w, c = z_perm.shape

        # (B,H*W,C)
        z_perm = z_perm.view(b, -1, c)
        # (B,H*W,C), (B,H*W)
        z_hat, indices, commit_loss = self.quantizer(z_perm)
        # (B,H,W,C)
        z_hat = z_hat.view(b, h, w, c)
        # (B,C,H,W)
        z_hat = z_hat.permute(0, 3, 1, 2)
        # (B,H,W)
        indices = indices.view(b, h, w)

        return HyperAlignOutput(z=z, z_hat=z_hat, indices=indices, commit_loss=commit_loss)
    
class HyperAlign_cor(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config 
    def __init__(
            self,
            in_channels: int = 4,
            N=192,
            M=320,
            codebook_dim=32,
            cfg_ss=64,
            cfg_cs=256,
    ):
        super().__init__()

        vq_spatialdim = cfg_ss
        self.backbone = nn.Sequential(
            # conv1x1(in_channels, N) if vq_spatialdim >= 16 else conv(in_channels, N),
            conv1x1(in_channels, N),
            # conv(in_channels, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            # conv1x1(N, N) if vq_spatialdim >= 32 else conv(N, N),
            conv1x1(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            # conv1x1(N, N) if vq_spatialdim == 64 else conv(N, N),
            conv1x1(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, M),
            AttentionBlock(M)
        )

        # VQ
        self.quantizer = VectorQuantize(dim=M,
                                        codebook_size=cfg_cs,
                                        use_cosine_sim = True,
                                        codebook_dim=codebook_dim)

    def forward(self, z: Tensor) -> Tensor:
        if torch.isnan(z).any():
            print("NaN detected in input!")
        if torch.isinf(z).any():
            print("Inf detected in input!")

        z = self.backbone(z)
        z_perm = z.permute(0, 2, 3, 1)
        b, h, w, c = z_perm.shape


        z_perm = z_perm.view(b, -1, c)
        z_hat, indices, commit_loss = self.quantizer(z_perm)

        z_hat = z_hat.view(b, h, w, c)

        z_hat = z_hat.permute(0, 3, 1, 2)

        indices = indices.view(b, h, w)

        return HyperAlignOutput(z=z, z_hat=z_hat, indices=indices, commit_loss=commit_loss)    
    
class CSIfusion(nn.Module):
    def __init__(self, dim=320, num_heads=8, depth=2):
        super().__init__()
        self.dim = dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.pos_embed = nn.Parameter(torch.randn(1, 10000, dim))  

        self.proj_out = nn.Linear(dim, dim)

    def forward(self, z_hat_cor, z_hat_target, common_mask):
        B, C, H, W = z_hat_cor.shape
        _, _, H1, W1 = z_hat_target.shape
        cor_flat = z_hat_cor.flatten(2).transpose(1, 2)       
        tgt_flat = z_hat_target.flatten(2).transpose(1, 2)      

        mask_flat = common_mask.flatten(1).unsqueeze(-1)       
        cor_common = cor_flat * mask_flat                       

        fused = torch.cat([cor_common, tgt_flat], dim=1)

        pos = self.pos_embed[:, :fused.shape[1], :]
        fused = fused + pos

        fused_out = self.transformer(fused) 
        fused_target = fused_out[:, -H1*W1:, :]
        fused_target = self.proj_out(fused_target)
        fused_target = fused_target.transpose(1, 2).reshape(B, C, H1, W1)

        return fused_target


class CommonPrior(nn.Module):

    def __init__(self, in_channels, hidden_dim=512, tau_init=5.0, tau_min=0.5, anneal_rate=0.99):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*4, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1)   
        )
        # Gumbel-Softmax 
        self.tau = tau_init
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate

    def gumbel_sigmoid(self, logits, tau=1.0, hard=False, eps=1e-10, thera=0.9):
        U = torch.rand_like(logits)
        g = -torch.log(-torch.log(U + eps) + eps)  # Gumbel noise
        y = torch.sigmoid((logits + g) / tau)      
        if hard:
            y_hard = (y > thera).float()
            y = y_hard + (y - y.detach())
        return y

    def forward(self, z_hat, z_hat_cor, hard=False,thera=0.9):
        if z_hat.size(2)!=z_hat_cor.size(2):
            z_hat = F.interpolate(z_hat, size=z_hat_cor.shape[2:], mode='nearest')
        diff = torch.abs(z_hat - z_hat_cor)
        prod = z_hat * z_hat_cor
        z_hat_cor = torch.cat([z_hat, z_hat_cor, diff, prod], dim=1)

        logits = self.conv(z_hat_cor)         
        mask = self.gumbel_sigmoid(logits, tau=self.tau, hard=hard,thera=thera)  
        mask = mask[:,0,:,:]            
        return mask

    def anneal_tau(self):
        self.tau = max(self.tau * self.anneal_rate, self.tau_min)

