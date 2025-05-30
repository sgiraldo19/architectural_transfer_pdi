import os
import itertools
import random
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
DATA_DIR = "data"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# DATASET
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform):
        self.files_A = glob(os.path.join(root_A, '*'))
        self.files_B = glob(os.path.join(root_B, '*'))
        self.transform = transform

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[index % len(self.files_B)]).convert("RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

# -----------------------------
# TRANSFORMACIONES
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# RESNET BLOCK
# -----------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

# -----------------------------
# GENERADOR
# -----------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        # Downsampling
        in_feat = 64
        out_feat = in_feat * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(True)
            ]
            in_feat = out_feat
            out_feat = in_feat * 2

        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ResnetBlock(in_feat)]

        # Upsampling
        out_feat = in_feat // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feat, out_feat, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(True)
            ]
            in_feat = out_feat
            out_feat = in_feat // 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# DISCRIMINADOR
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        ]
        in_feat = 64
        out_feat = in_feat * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.LeakyReLU(0.2)
            ]
            in_feat = out_feat
            out_feat = in_feat * 2

        model += [nn.Conv2d(in_feat, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# ENTRENAMIENTO
# -----------------------------
def train():
    # Cargar datos
    dataset = ImageDataset(f"{DATA_DIR}/A", f"{DATA_DIR}/B", transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instanciar modelos
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Optimización
    g_optimizer = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=2e-4, betas=(0.5, 0.999))

    # Funciones de pérdida
    adversarial_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(loop):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            valid = torch.ones((real_A.size(0), 1, 30, 30), device=device)
            fake = torch.zeros((real_A.size(0), 1, 30, 30), device=device)

            # -------- Generadores --------
            g_optimizer.zero_grad()

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            loss_id_A = identity_loss(G_BA(real_A), real_A)
            loss_id_B = identity_loss(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) * 0.5

            loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
            loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)

            recovered_A = G_BA(fake_B)
            recovered_B = G_AB(fake_A)
            loss_cycle = (cycle_loss(recovered_A, real_A) + cycle_loss(recovered_B, real_B)) * 10

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle + loss_identity
            loss_G.backward()
            g_optimizer.step()

            # -------- Discriminadores --------
            d_optimizer.zero_grad()
            loss_D_A = adversarial_loss(D_A(real_A), valid) + adversarial_loss(D_A(fake_A.detach()), fake)
            loss_D_B = adversarial_loss(D_B(real_B), valid) + adversarial_loss(D_B(fake_B.detach()), fake)
            loss_D = (loss_D_A + loss_D_B) * 0.5
            loss_D.backward()
            d_optimizer.step()

            loop.set_postfix(G=loss_G.item(), D=loss_D.item())

        # Guardar imágenes generadas por época
        with torch.no_grad():
            fakes = G_AB(real_A)
            save_image((fakes * 0.5 + 0.5), f"{SAVE_DIR}/epoch_{epoch+1}.png", nrow=2)

        # Guardar modelos
        torch.save(G_AB.state_dict(), f"{SAVE_DIR}/G_AB_{epoch+1}.pth")
        torch.save(G_BA.state_dict(), f"{SAVE_DIR}/G_BA_{epoch+1}.pth")

if __name__ == "__main__":
    train()
