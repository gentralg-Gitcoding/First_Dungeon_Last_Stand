from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import numpy as np

#User files
from settings import GAN_TO_ROOM_TILE, MATRIX_TO_ROOM_TILE, ROOM_WIDTH, ROOM_HEIGHT, ROOM_TYPES


class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_room_types=5):
        super().__init__()

        self.embedding = nn.Embedding(num_room_types, 16)

        self.fc = nn.Linear(noise_dim + 16, 128 * 6 * 10)   # Initial fully connected layer to expand noise + room type embedding into a feature map

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 6, kernel_size=3, padding=1),  # 6 channels for 6 tile types  
        )

    def forward(self, z, room_type):
        emb = self.embedding(room_type)                                         # shape: (batch, 16)
        x = torch.cat([z, emb], dim=1)                                          # concatenate noise and room type embedding
        x = self.fc(x)                                                          # shape: (batch, 128*6*10)
        x = x.view(-1, 128, 6, 10)                                              # Reshape to (batch, 128, 6, 10)
        x = self.net(x)                                                         # shape: (batch, 6, 16, 16)
        x = nn.functional.interpolate(x, size=(ROOM_HEIGHT, ROOM_WIDTH))        # Resize to EXACT dungeon size
        # x = nn.functional.softmax(x, dim=1)                                     # Softmax across tile type channels   
        return x

class Discriminator(nn.Module):
    def __init__(self, num_room_types=5):
        super().__init__()

        self.embedding = nn.Embedding(num_room_types, 22 * 40)     #Embed room type to a full matrix that can be concatenated with the room input

        self.net = nn.Sequential(
            nn.Conv2d(6 + 1, 32, kernel_size=4, stride=2, padding=1),   # 6 channels for tile types + 1 for room type embedding
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(64 * 5 * 10, 1),      # Adjust based on final feature map size
            nn.Sigmoid()
        )

    def forward(self, x, room_type):
        batch_size = x.size(0)

        emb = self.embedding(room_type)
        emb = emb.view(batch_size, 1, ROOM_HEIGHT, ROOM_WIDTH)   # Reshape to (batch, 1, height, width)

        # emb = self.embedding(room_type)              # (B, embed_dim)
        # emb = emb.unsqueeze(-1).unsqueeze(-1)        # (B, embed_dim, 1, 1)
        # emb = emb.expand(-1, -1, ROOM_HEIGHT, ROOM_WIDTH)  # (B, embed_dim, H, W)

        x = torch.cat([x, emb], dim=1)      # Concatenate along channel dimension

        return self.net(x)

def train_gan(generator, discriminator, dataloader, epochs=50, device='cpu'):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)
    discriminator.to(device)

    bce_criterion = nn.BCELoss()
    class_weights = get_imbalanced_class_weights(dataloader.dataset).to(device)
    class_weights = torch.log(class_weights + 1)        #Prevent extreme weights that could destabilize training
    print(f"Using class weights: {class_weights}")
    ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    g_opt = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=0.00005)  # Lower LR for discriminator to prevent overpowering generator

    for epoch in range(epochs):
        for real_rooms, room_types  in dataloader:

            real_rooms = real_rooms.to(device)
            room_types = room_types.to(device)
            batch_size = real_rooms.size(0)

            # REAL labels = 1, FAKE = 0
            # real_labels = torch.ones(batch_size, 1).to(device)
            # fake_labels = torch.zeros(batch_size, 1).to(device)

            #Label smoothing: instead of 1s and 0s, use 0.9 and 0.1 to prevent discriminator from becoming too confident
            real_labels = torch.full((batch_size, 1), 0.9).to(device)
            fake_labels = torch.full((batch_size, 1), 0.1).to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            z = torch.randn(batch_size, 100).to(device)

            fake_rooms = generator(z, room_types)

            #Add noise to real and fake rooms to make discriminator more robust and prevent overfitting
            noise = torch.randn_like(real_rooms) * 0.05
            real_rooms_noisy = real_rooms + noise

            noise = torch.randn_like(fake_rooms) * 0.05
            fake_rooms_noisy = fake_rooms + noise

            d_real = discriminator(real_rooms_noisy, room_types)
            d_fake = discriminator(fake_rooms_noisy.detach(), room_types)

            d_loss = bce_criterion(d_real, real_labels) + bce_criterion(d_fake, fake_labels)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # -----------------
            # Train Generator
            # -----------------
            for _ in range(2):
                z = torch.randn(batch_size, 100).to(device)
                fake_rooms = generator(z, room_types)   # (B, 6, H, W)

                # ----------------------
                # Combined adversarial loss with pixel-wise classification loss to encourage generated rooms 
                # to not only fool the discriminator but also have correct tile distributions
                # ----------------------
                target = torch.argmax(real_rooms, dim=1)  # (B, H, W)
                ce_loss = ce_criterion(fake_rooms, target)

                g_loss = bce_criterion(discriminator(fake_rooms, room_types), real_labels)
                g_loss = g_loss + 0.05 * ce_loss  # Combine adversarial loss with pixel-wise classification loss

                # ----------------------
                # Entropy regularization to encourage more diverse tile distributions in generated rooms
                # ----------------------
                probs = torch.softmax(fake_rooms, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                g_loss = g_loss - 0.01 * entropy

                # # ---------------------
                # # Room type embedding regularization to encourage more meaningful room type representations
                # # ---------------------
                # type_embedding = generator.embedding(room_types)
                # type_strength = type_embedding.norm(dim=1).mean()

                # g_loss = g_loss + 0.01 * type_strength  # Encourage stronger room type embeddings

                # # ---------------------
                # # Entity density penalty to encourage more interesting rooms, but not too much
                # # ---------------------
                # entity_probs = torch.softmax(fake_rooms, dim=1)

                # enemy_density = entity_probs[:, 1].mean()
                # chest_density = entity_probs[:, 2].mean()
                # heal_density = entity_probs[:, 3].mean()

                # target_density = 0.15  # Target density for entities in a room, can be tuned
                # entity_density = enemy_density + chest_density + heal_density
                # density_penalty = torch.abs(entity_density - target_density)

                # g_loss = g_loss + 0.1 * density_penalty    # Encourage more entities overall, but not too much

                # # ---------------------
                # # Reward for empty space to prevent overcrowding
                # # ---------------------
                # empty_prob = entity_probs[:, 0].mean()

                # g_loss = g_loss - 0.1 * empty_prob

                # # ---------------------
                # # Penalty for clustering of entities to encourage more natural room layouts
                # # ---------------------
                # enemy_map = entity_probs[:, 1]


                # # Look in 4 directions and penalize if neighboring tiles also have high enemy probability (indicating clustering)
                # cluster_loss = (
                #     torch.abs(enemy_map - torch.roll(enemy_map, 1, 1)).mean() +
                #     torch.abs(enemy_map - torch.roll(enemy_map, -1, 1)).mean() +
                #     torch.abs(enemy_map - torch.roll(enemy_map, 1, 2)).mean() +
                #     torch.abs(enemy_map - torch.roll(enemy_map, -1, 2)).mean()
                # )

                # g_loss = g_loss + 0.2 * cluster_loss

                # # ---------------------
                # # Penalty for center bias spawning
                # # ---------------------
                # H, W = ROOM_HEIGHT, ROOM_WIDTH
                # y_coords = torch.linspace(-1, 1, H).view(1, H, 1).to(device)
                # x_coords = torch.linspace(-1, 1, W).view(1, 1, W).to(device)

                # distance = torch.sqrt(x_coords**2 + y_coords**2)

                # center_weight = 1 - distance  # center = high value

                # center_loss = -(entity_probs[:, 1] * center_weight).mean()

                # g_loss = g_loss + 0.1 * center_loss



                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

        # -----------------
        # Visualization 
        # -----------------
        # visualize_generator(generator, device, epoch)

        # -----------------
        # Save models every 10 epochs
        # -----------------
        if epoch % 10 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), f"game/data/models/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"game/data/models/discriminator_epoch_{epoch}.pth")

        print(f"Epoch {epoch}: D={d_loss.item():.4f}, G={g_loss.item():.4f}")


def generate_room(generator, room_type_str, width, height):
    generator.eval()

    #Dont modify start and boss rooms
    if room_type_str in ["start", "boss"]:
        return None

    device = next(generator.parameters()).device

    #Turn roomtype into tensor
    room_type = torch.tensor([ROOM_TYPES[room_type_str]], dtype=torch.long, device=device)

    with torch.no_grad():
        z = torch.randn(1, 100, device=device)
        room_matrix = generator(z, room_type)   # shape: (1, 4, height, width) - probabilities for each tile type at each position

        room_matrix = torch.argmax(room_matrix, dim=1)      # shape: (1, height, width) - tile type with highest probability at each position

        room_matrix = room_matrix.squeeze(0).cpu().numpy()  # shape: (height, width) - convert to numpy array
    print(f"Generated room of type {room_type_str} with shape {room_matrix.shape}")
    unique, counts = np.unique(room_matrix, return_counts=True)
    print(dict(zip(unique, counts)))
    return room_matrix


def tensor_to_room_tile(tensor):
    """
    tensor: (6, H, W) OR (1, 6, H, W)
    """

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tile_indices = torch.argmax(tensor, dim=0)  # (H, W)

    room = []
    for row in tile_indices:
        line = "".join(MATRIX_TO_ROOM_TILE[int(tile)] for tile in row)
        room.append(line)

    return "\n".join(room)


def visualize_generator(generator, device, epoch, num_samples=3):
    '''
    visualize training progress by generating sample rooms for each room type and printing them as ASCII art
    '''
    generator.eval()

    print(f"\n=== Epoch {epoch} Samples ===")

    for room_type_str, room_type_id in ROOM_TYPES.items():

        # Skip start/boss if you want
        # if room_type_str in ["start", "boss"]:
        #     continue

        print(f"\n--- {room_type_str.upper()} ROOM ---")

        for _ in range(num_samples):
            z = torch.randn(1, 100).to(device)
            room_type = torch.tensor([room_type_id]).to(device)

            with torch.no_grad():
                output = generator(z, room_type)

            tile_room = tensor_to_room_tile(output)

            print(tile_room)
            print("-" * 40)

def get_imbalanced_class_weights(dataset):
    # Flatten ALL tiles from dataset
    all_tiles = []

    for room_tensor, _ in dataset:
        class_map = torch.argmax(room_tensor, dim=0)
        flat = np.array(class_map).flatten()
        all_tiles.extend(flat)

    classes = np.unique(all_tiles)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=all_tiles
    )

    class_weights = torch.tensor(weights, dtype=torch.float32)
    return class_weights