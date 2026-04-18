import torch

# User files
from ai.diffusion_generator import SimpleUNet, train_diffusion_model
from utils.data_to_dataloader_converter import get_dataloader

DATA_PATH = "game/data/diffusion_tensors.npz"

ROOM_TYPES = {
    "enemy": 0,
    "loot": 1,
    "healing": 2,
    "start": 3,
    "boss": 4,
}

def get_noise_schedule(T=200, device="cpu"):
    beta_start = 1e-4
    beta_end = 0.02

    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return alphas_cumprod


def main():
    # -----------------------
    # Device
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Load Data
    # -----------------------
    dataloader = get_dataloader(DATA_PATH, batch_size=32)

    # -----------------------
    # Initialize Models
    # -----------------------
    generator = SimpleUNet(in_channels=6, num_room_types=len(ROOM_TYPES)).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    # -----------------------
    # Noise Schedule
    # -----------------------
    alphas_cumprod = get_noise_schedule(device=device)

    # -----------------------
    # Train
    # -----------------------
    train_diffusion_model(
        model=generator,
        dataloader=dataloader,
        optimizer=optimizer,
        alphas_cumprod=alphas_cumprod,
        epochs=50,
        device=device,
    )
    print(f"Used device: {device}")


if __name__ == "__main__":
    main()