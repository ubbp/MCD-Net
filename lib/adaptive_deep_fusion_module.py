import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class Residual(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ADFBlock(nn.Module):
    def __init__(self, dim, channels, mlp_ratio=2):
        super().__init__()
       

        self.norm1 = nn.LayerNorm(dim)
        self.tma = SpatialImageLanguageAttention(dim, 768, dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, 2*dim)

    def forward(self, x, l, l_mask):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        out = self.norm1(x)
        out = x + self.tma(out, l, l_mask)

        out = out + self.ff(self.norm2(out))
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out



class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class FrequencyEnhance(nn.Module):
    def __init__(self, enhance_ratio=1.5, split_threshold=0.2):
        super(FrequencyEnhance, self).__init__()
        self.enhance_ratio = enhance_ratio
        self.split_threshold = split_threshold

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            enhanced x
        """
        B, C, H, W = x.shape
        x_freq = torch.fft.fft2(x, norm='ortho')  # (B, C, H, W), complex tensor
        mask = self._generate_frequency_mask(H, W, self.split_threshold, x.device)
        low_freq = x_freq * mask
        high_freq = x_freq * (1 - mask)
        enhanced_high_freq = high_freq * self.enhance_ratio
        enhanced_freq = low_freq + enhanced_high_freq
        x_rec = torch.fft.ifft2(enhanced_freq, norm='ortho').real 
        return x_rec

    def _generate_frequency_mask(self, H, W, threshold, device):
    
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        freq_radius = torch.sqrt(xx ** 2 + yy ** 2)
        mask = (freq_radius <= threshold).float()
        return mask.unsqueeze(0).unsqueeze(0) 
   
class ScaleAwareGate(nn.Module):
    def __init__(self, inp, oup, sam_inp):
        super(ScaleAwareGate, self).__init__()

        self.local_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(oup)

        self.global_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(oup)

        self.global_act = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(oup)
        self.act = h_sigmoid()
        self.act_withsam = h_sigmoid()
        self.local_embedding_sam = nn.Conv2d(sam_inp, oup, kernel_size=1)
        self.bn_sam = nn.BatchNorm2d(oup)

        self.local_act_sam = nn.Conv2d(sam_inp, oup, kernel_size=1)
        self.bn_act_sam = nn.BatchNorm2d(oup)

    def forward(self, x_l, x_g, x_sam):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        local_feat = self.bn1(local_feat)
        global_feat = self.global_embedding(x_g)
        global_feat = self.bn2(global_feat)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        global_act = self.global_act(x_g)
        global_act = self.bn3(global_act)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)  
        sam_feat = self.local_embedding_sam(x_sam)
        sam_feat = self.bn_sam(sam_feat)
        sam_feat = F.interpolate(sam_feat, size=(H, W), mode='bilinear', align_corners=False)

        act_withsam = self.local_act_sam(x_sam)
        act_withsam = self.bn_act_sam(act_withsam)
        sig_act_withsam = F.interpolate(self.act_withsam(act_withsam), size=(H, W), mode='bilinear', align_corners=False) 

        out = local_feat * sig_act + global_feat + sam_feat * sig_act_withsam
        return out
       
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x  
    
class ADF(nn.Module):
    def __init__(self, dim, num_blocks=1, channels=[128, 256, 512, 1024], downsample=1):
        super().__init__()
        self.hidden_dim = dim // 4
        self.channels = channels
        self.stride = downsample

        self.down_channel = nn.Conv2d(dim, self.hidden_dim, 1)
        self.up_channel = nn.Conv2d(self.hidden_dim, dim, 1)

        self.pool = PyramidPoolAgg(stride=self.stride)
        self.block = nn.ModuleList([
            ADFBlock(self.hidden_dim, channels)
            for _ in range(num_blocks)
        ])
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        
        self.fusion_withsam = nn.ModuleList([
            ScaleAwareGate(channels[i], channels[i], channels[i])
            for i in range(len(channels))
        ])

        self.fusion_withsam2 = ScaleAwareGate(channels[-1], channels[-1], 256)
            

        self.neck = nn.ModuleList([nn.Sequential(
            nn.Conv2d(
                384,
                channels[i],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(channels[i]),
            nn.Conv2d(
                channels[i],
                channels[i],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(channels[i]),
        )
        for i in range(len(channels)-1)])

    def forward(self, input, input_sam_list, l, l_mask):  
       
        out = self.pool(input)
        out = self.down_channel(out)
        new_input_sam_list = [] 
  
        for j in range(len(input_sam_list)-1):
            
            input_sam = input_sam_list[j].reshape(input_sam_list[j].shape[0], 64, 64, 384)
            
            input_sam = self.neck[j](input_sam.permute(0, 3, 1, 2))
            new_input_sam_list.append(input_sam)
        new_input_sam_list.append(input_sam_list[-1])

        for layer in self.block:
            out = layer(out, l, l_mask)
        out = self.bn(out)
        out = self.up_channel(out)
        xx = out.split(self.channels, dim=1)

        freq_enhance = FrequencyEnhance(enhance_ratio=1.5, split_threshold=0.2)

        results = []
        
        for i in range(len(self.channels)-1):
            ADF_l = input[i]
            ADF_g = xx[i]
            new_input_sam = new_input_sam_list[i]

            ADF_l = freq_enhance(ADF_l)
            ADF_g = freq_enhance(ADF_g)
            
            out_ = self.fusion_withsam[i](ADF_l, ADF_g, new_input_sam)
            results.append(out_)
            

        ADF_l = input[-1]
        ADF_g = xx[-1]
        new_input_sam = new_input_sam_list[-1]
        out_ = self.fusion_withsam2(ADF_l, ADF_g, new_input_sam)
        results.append(out_)
        

        
        return results


