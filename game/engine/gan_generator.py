import torch
import torch.nn as nn

#User files
from settings import *

class Generator(nn.Module):
    # def __init__(self, noise_dim=100, room_size=16):
    def __init__(self, noise_dim=100):
        super().__init__()

        self.room_width = ROOM_WIDTH
        self.room_height = ROOM_HEIGHT

        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.room_width * self.room_height),
            nn.Tanh()       #GANs prefer centered distributions, so we use Tanh to center around 0
        )

        # self.room_size = room_size

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1, self.room_height, self.room_width)
        return x

class Discriminator(nn.Module):
    # def __init__(self, room_size=16):
    def __init__(self):
        super().__init__()

        self.room_width = ROOM_WIDTH
        self.room_height = ROOM_HEIGHT

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.room_width * self.room_height, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, dataloader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()

    g_opt = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_rooms in dataloader:

            real_rooms = real_rooms.to(device)
            batch_size = real_rooms.size(0)

            # REAL labels = 1, FAKE = 0
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            z = torch.randn(batch_size, 100).to(device)
            fake_rooms = generator(z)

            d_loss_real = criterion(discriminator(real_rooms), real_labels)
            d_loss_fake = criterion(discriminator(fake_rooms.detach()), fake_labels)

            d_loss = d_loss_real + d_loss_fake

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # -----------------
            # Train Generator
            # -----------------
            z = torch.randn(batch_size, 100).to(device)
            fake_rooms = generator(z)

            g_loss = criterion(discriminator(fake_rooms), real_labels)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        print(f"Epoch {epoch}: D={d_loss.item():.4f}, G={g_loss.item():.4f}")

def generate_room(generator, width, height):
    generator.eval()

    with torch.no_grad():
        z = torch.randn(1, 100)
        room = generator(z).squeeze().cpu().numpy()

    # Convert from tanh's [-1,1] → tile integers
    room = ((room + 1) / 2) * 5
    room = room.astype(int)

    return room[:height, :width]