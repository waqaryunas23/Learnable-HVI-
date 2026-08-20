import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
from main import OriginalHVI, LearnHVI  
from PIL import Image

class SimpleEnhancer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))  # keep output in [0,1]

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_paths = sorted([
            os.path.join(low_dir, f) for f in os.listdir(low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        self.high_paths = sorted([
            os.path.join(high_dir, f) for f in os.listdir(high_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = self.transform(Image.open(self.low_paths[idx]).convert("RGB"))
        high = self.transform(Image.open(self.high_paths[idx]).convert("RGB"))
        return low, high
    
def train_loop(hvi_module, model, dataloader, optimizer, device):
    model.train()
    criterion = nn.L1Loss()

    for epoch in range(10):
        for i, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            hvi_feat = hvi_module(low)
            output = model(hvi_feat)
            loss = criterion(output, high)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} Iter {i}: Loss={loss.item():.4f}")
                save_image(torch.cat([low, output, high], dim=0), f"vis_epoch{epoch}_iter{i}.png")

def evaluate(model, hvi_module, dataloader, device):
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            hvi_feat = hvi_module(low)
            output = model(hvi_feat)
            out_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            high_np = high.cpu().squeeze(0).permute(1, 2, 0).numpy()

            psnr_list.append(psnr(high_np, out_np, data_range=1))
            ssim_list.append(ssim(high_np, out_np, data_range=1, channel_axis=2))

    print(f"Avg PSNR: {sum(psnr_list)/len(psnr_list):.2f}, SSIM: {sum(ssim_list)/len(ssim_list):.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_dataset = PairedImageDataset(r"D:\learniable_hvi\LOLdataset (2)\our485\low", r"D:\learniable_hvi\LOLdataset (2)\our485\high")
V_dataset = PairedImageDataset(r"D:\learniable_hvi\LOLdataset (2)\eval15\low",r"D:\learniable_hvi\LOLdataset (2)\eval15\high")
loader = DataLoader(t_dataset, batch_size=16, shuffle=True)
v_loader = DataLoader(V_dataset,batch_size=1, shuffle=False)

# Try original HVI
hvi = OriginalHVI().to(device)
model = SimpleEnhancer(3, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#training hsvi
train_loop(hvi, model, loader, optimizer, device)
evaluate(model, hvi, v_loader, device)

# try LearnHVI
hvi_learned = LearnHVI().to(device)
model_learn = SimpleEnhancer(3, 3).to(device)
optimizer_learn = torch.optim.Adam(list(model_learn.parameters()) + list(hvi_learned.parameters()), lr=1e-4)

train_loop(hvi_learned, model_learn, loader, optimizer_learn, device)
evaluate(model_learn, hvi_learned, v_loader, device)

