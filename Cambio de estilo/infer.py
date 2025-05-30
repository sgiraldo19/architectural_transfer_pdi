import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn

# -----------------------------
# Modelo Generator (igual que el del entrenamiento)
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
# Función para cargar un modelo desde archivo
# -----------------------------
def load_generator(path, device='cuda'):
    model = Generator()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

# -----------------------------
# Transformación estándar de imagen
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# Función de inferencia (usa modelo ya cargado)
# -----------------------------
def run_inference(image_path, model, output_path, device='cuda'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = (output_tensor.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    output_image.save(output_path)

    print(f"Imagen generada guardada en: {output_path}")

# -----------------------------
# CLI (opcional) para uso desde terminal
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia con Generator de CycleGAN")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen de entrada")
    parser.add_argument("--model", type=str, required=True, help="Ruta del modelo entrenado (.pth)")
    parser.add_argument("--output", type=str, default="output.jpg", help="Ruta para guardar imagen generada")
    parser.add_argument("--cpu", action="store_true", help="Usar CPU en vez de GPU")

    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_generator(args.model, device)
    run_inference(args.image, model, args.output, device)
