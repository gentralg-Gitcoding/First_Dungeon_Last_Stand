import torch

# User files
from engine.gan_generator import Generator, Discriminator, train_gan
from utils.data_to_dataloader_converter import get_dataloader

DATA_PATH = "game/data/synthetic_rooms_tensors.npz"


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
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # -----------------------
    # Train
    # -----------------------
    train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        epochs=50,
        device=device,
        # visualize_fn=visualize_generator  
    )
    print(f"Used device: {device}")


if __name__ == "__main__":
    main()