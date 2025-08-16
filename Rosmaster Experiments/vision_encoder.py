import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalGlobalBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        # Dynamic local kernel generation
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.kernel_gen = nn.Sequential(
            nn.Linear(mid_channels, mid_channels * 3 * 3, bias=False),
            nn.ReLU()
        )
        # Global stream with dilation
        self.global_stream = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        # Downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Split into local & global channels
        local_input, global_input = x.chunk(2, dim=1)
        # Dynamic local convolution per sample
        b, c, h, w = local_input.shape
        pooled = self.pool(local_input).view(b, c)
        kernels = self.kernel_gen(pooled).view(b * c, 1, 3, 3)
        # Reshape for grouped conv
        local_reshaped = local_input.reshape(1, b * c, h, w)
        local_out = F.conv2d(
            local_reshaped,
            weight=kernels,
            bias=None,
            stride=1,
            padding=1,
            groups=b * c
        )
        local_out = local_out.view(b, c, h, w)
        # Global dilated conv
        global_out = self.global_stream(global_input)
        # Combine and downsample
        combined = torch.cat([local_out, global_out], dim=1)
        out = self.downsample(combined)
        return out

class CrossScaleGate(nn.Module):
    def __init__(self, in_channels=[12, 16, 24], reduced_dim=16):
        super().__init__()
        self.total_channels = sum(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gating = nn.Sequential(
            nn.Linear(self.total_channels, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, self.total_channels),
            nn.Softmax(dim=1)
        )
        nn.init.kaiming_normal_(self.gating[0].weight)
        nn.init.kaiming_normal_(self.gating[2].weight)

    def forward(self, high_res_map, med_res_map, low_res_map):
        xx = high_res_map.shape[-1]
        med_res_up = F.interpolate(med_res_map, size=(xx, xx), mode="bilinear", align_corners=False)
        low_res_up = F.interpolate(low_res_map, size=(xx, xx), mode="bilinear", align_corners=False)
        fused = torch.cat([high_res_map, med_res_up, low_res_up], dim=1)
        pooled_fused = self.pool(fused).squeeze(-1).squeeze(-1)
        weights = self.gating(pooled_fused)
        weights = weights.view(fused.size(0), self.total_channels, 1, 1)
        fused_weighted = fused * weights
        return fused_weighted

class AdaptScaleInjection(nn.Module):
    def __init__(self, in_channels=[12, 16, 24]):
        super().__init__()
        # Channel attention
        self.high_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[0], in_channels[0] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels[0] // 4, in_channels[0], kernel_size=1),
            nn.Sigmoid()
        )
        self.med_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[1], in_channels[1] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels[1] // 4, in_channels[1], kernel_size=1),
            nn.Sigmoid()
        )
        self.low_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[2], in_channels[2] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels[2] // 4, in_channels[2], kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial attention convs with scale-specific kernels
        self.high_spatial = nn.Conv2d(3, 1, kernel_size=3, padding=1)  # Increased to 3 inputs for gradient magnitude
        self.med_spatial  = nn.Conv2d(3, 1, kernel_size=5, padding=2)  # Increased to 3 inputs for local variance
        self.low_spatial  = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        # Register Sobel kernels for edge detection
        self.register_buffer('sobel_x_kernel', torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32))
        self.register_buffer('sobel_y_kernel', torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32))
        
        for sa in [self.high_spatial, self.med_spatial, self.low_spatial]:
            nn.init.kaiming_normal_(sa.weight, nonlinearity='relu')

    def forward(self, high_res, med_res, low_res):
        h_c = high_res * self.high_channel_attn(high_res)
        m_c = med_res  * self.med_channel_attn(med_res)
        l_c = low_res  * self.low_channel_attn(low_res)
        
        # High-res spatial attention with gradient magnitude for edges
        avg_h = torch.mean(h_c, dim=1, keepdim=True)
        mx_h, _ = torch.max(h_c, dim=1, keepdim=True)
        grad_x_h = F.conv2d(avg_h, self.sobel_x_kernel, padding=1)
        grad_y_h = F.conv2d(avg_h, self.sobel_y_kernel, padding=1)
        grad_mag_h = torch.sqrt(grad_x_h**2 + grad_y_h**2 + 1e-6)
        h_sa_input = torch.cat([avg_h, mx_h, grad_mag_h], dim=1)
        h_sa = torch.sigmoid(self.high_spatial(h_sa_input))
        
        # Med-res spatial attention with local variance for traffic signs
        avg_m = torch.mean(m_c, dim=1, keepdim=True)
        mx_m, _ = torch.max(m_c, dim=1, keepdim=True)
        local_mean_m = F.avg_pool2d(avg_m, kernel_size=3, stride=1, padding=1)
        local_squared_mean_m = F.avg_pool2d(avg_m**2, kernel_size=3, stride=1, padding=1)
        local_var_m = local_squared_mean_m - local_mean_m**2 + 1e-6
        m_sa_input = torch.cat([avg_m, mx_m, local_var_m], dim=1)
        m_sa = torch.sigmoid(self.med_spatial(m_sa_input))
        
        
        # Low-res spatial attention (unchanged)
        avg_l = torch.mean(l_c, dim=1, keepdim=True)
        mx_l, _ = torch.max(l_c, dim=1, keepdim=True)
        l_sa_input = torch.cat([avg_l, mx_l], dim=1)
        l_sa = torch.sigmoid(self.low_spatial(l_sa_input))
        
        h_out = h_c * h_sa
        m_out = m_c * m_sa
        l_out = l_c * l_sa
        return h_out, m_out, l_out, h_sa, m_sa, l_sa

class MultiScaleVisionEncoder(nn.Module):
    def __init__(self, stem_channels=12, branch_channels=[12, 16, 24], num_classes=11):
        super().__init__()
        self.num_stem_channels = stem_channels
        self.num_high_res_channels = branch_channels[0]
        self.num_med_res_channels = branch_channels[1]
        self.num_low_res_channels = branch_channels[2]
        self.proj_channels = 8
        self.total_channels = 3 * self.proj_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.num_stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_stem_channels),
            nn.ReLU(inplace=True)
        )
        self.high_res_branch = nn.Sequential(
            nn.Conv2d(self.num_stem_channels, self.num_high_res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_high_res_channels),
            nn.ReLU(inplace=True)
        )
        self.med_res_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.num_stem_channels, self.num_med_res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_med_res_channels),
            nn.ReLU(inplace=True)
        )
        self.low_res_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.num_stem_channels, self.num_low_res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_low_res_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.high_res_proj = nn.Conv2d(self.num_high_res_channels, self.proj_channels, kernel_size=1)
        self.med_res_proj = nn.Conv2d(self.num_med_res_channels, self.proj_channels, kernel_size=1)
        self.low_res_proj = nn.Conv2d(self.num_low_res_channels, self.proj_channels, kernel_size=1)
        self.asi = AdaptScaleInjection(in_channels=[self.proj_channels, self.proj_channels, self.proj_channels])
        self.csgm = CrossScaleGate(in_channels=[self.proj_channels, self.proj_channels, self.proj_channels], reduced_dim=16)
        self.block1 = LocalGlobalBlock(in_channels=self.total_channels, mid_channels=self.total_channels // 2, out_channels=self.total_channels // 2)
        self.pre_pool = nn.Sequential(
            nn.Conv2d(self.total_channels // 2, self.total_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.total_channels // 4),
            nn.ReLU(),
            nn.Conv2d(self.total_channels // 4, self.total_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.total_channels // 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        xx = self.total_channels // 4
        self.FC = nn.Sequential(
            nn.Linear(xx*7*7, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        out = self.stem(x)
        high_res_map = self.high_res_branch(out)
        med_res_map = self.med_res_branch(out)
        low_res_map = self.low_res_branch(out)
        high_res_map = self.high_res_proj(high_res_map)
        med_res_map = self.med_res_proj(med_res_map)
        low_res_map = self.low_res_proj(low_res_map)
        high_res_map, med_res_map, low_res_map, attn_high, attn_med, attn_low = self.asi(high_res_map, med_res_map, low_res_map)
        #attn_high = attn_med = attn_low = None
        fused = self.csgm(high_res_map, med_res_map, low_res_map)
        
        #xx = high_res_map.shape[-1]
        #med_res_map_up = F.interpolate(med_res_map, size=(xx, xx), mode="bilinear", align_corners=False)
        #low_res_map_up = F.interpolate(low_res_map, size=(xx, xx), mode="bilinear", align_corners=False)
        #fused = torch.cat([high_res_map, med_res_map_up, low_res_map_up], dim=1)
        

        vision_embeddings = nn.AdaptiveAvgPool2d((1,1))(fused).squeeze(-1).squeeze(-1)
        fused = self.block1(fused)
        fused = self.pre_pool(fused)
        fused_flat = self.flat(fused)
        class_logits = self.FC(fused_flat)


        return class_logits, attn_high, attn_med, attn_low, vision_embeddings