import torch
import torch.nn as nn

pi = 3.14159265358979323846

def compute_hue(img, value, img_min, eps=1e-8):
    device = img.device
    dtypes = img.dtype
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
    value = img.max(1)[0].to(dtypes)
    img_min = img.min(1)[0].to(dtypes)
    hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
    hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
    hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

    hue[img.min(1)[0]==value] = 0.0
    hue = hue/6.0
    return hue.unsqueeze(1)

class LearnableIntensity(nn.Module):
    def __init__(self):
        super(LearnableIntensity, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, intensity):
        # intensity: (B, 1, H, W)
        B, C, H, W = intensity.shape
        x = intensity.view(B, -1, 1)  # Flatten and keep last dim for MLP
        y = self.mlp(x)
        return y.view(B, 1, H, W)

class LearnHVI(nn.Module):
    def __init__(self):
        super(LearnHVI, self).__init__()
        self.intensity_mapper = LearnableIntensity()
    
    def forward(self, img):
        eps = 1e-8
        value = img.max(1, keepdim=True)[0]
        img_min = img.min(1, keepdim=True)[0]
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0
        
        hue = compute_hue(img, value, img_min, eps)  # use your existing logic
        color_sensitive = self.intensity_mapper(value)

        ch = torch.cos(2 * pi * hue)
        cv = torch.sin(2 * pi * hue)
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        
        return torch.cat([H, V, I], dim=1)

"""class OriginalHVI(nn.Module):
    def __init__(self, density_k=0.2):
        super().__init__()
        self.k = density_k

    def forward(self, img):
        eps = 1e-8
        value = img.max(1, keepdim=True)[0]
        img_min = img.min(1, keepdim=True)[0]
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0
        
        hue = compute_hue(img, value, img_min, eps)  # you’ll reuse this
        color_sensitive = ((torch.sin(value * 0.5 * pi) + eps).pow(self.k))
        
        ch = torch.cos(2 * pi * hue)
        cv = torch.sin(2 * pi * hue)
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        
        return torch.cat([H, V, I], dim=1)"""
