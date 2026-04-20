import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from settings import ROOM_TILE_DICT

DIFFUSION_PATH = "game/data/models/diffusion_model.pth"
NUM_CHANNELS = 6

WALL = ROOM_TILE_DICT['WALL']
FLOOR = ROOM_TILE_DICT['FLOOR']
DOOR = ROOM_TILE_DICT['DOOR']
ENEMY = ROOM_TILE_DICT['ENEMY']
CHEST = ROOM_TILE_DICT['CHEST']
HEALING = ROOM_TILE_DICT['HEALING']

CHANNEL_MAP = {
    WALL: 0,
    FLOOR: 1,
    DOOR: 2,
    ENEMY: 3,
    CHEST: 4,
    HEALING: 5
}

# --- Time embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        t = t.float().unsqueeze(-1)  # (B, 1)
        return F.relu(self.linear2(F.relu(self.linear1(t))))


# --- Block with time conditioning ---
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x))
        
        # Inject time embedding
        time_term = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term

        h = F.relu(self.conv2(h))
        return h


# --- Simple UNet ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=6, num_room_types=5, time_dim=128):    # in_channels = 6 tiles + 5 room types
        super().__init__()

        self.num_room_types = num_room_types
        total_in_channels = in_channels + num_room_types

        self.time_embed = TimeEmbedding(time_dim)
        self.type_embedding = nn.Embedding(num_room_types, time_dim)

        # Down
        self.b1 = Block(total_in_channels, 64, time_dim)
        self.b2 = Block(64, 128, time_dim)

        # Up
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.b3 = Block(128, 64, time_dim)  # skip connection

        self.out = nn.Conv2d(64, in_channels, 1)    # Keep output same as original channels

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t, room_type):
        t_emb = self.time_embed(t)
        type_emb = self.type_embedding(room_type)
        combined_emb = t_emb + type_emb

        #Type channel input
        type_channel = F.one_hot(room_type, num_classes=self.num_room_types).float()    # (B, T)
        type_channel = type_channel.unsqueeze(-1).unsqueeze(-1)                         # (B, T, 1, 1)
        type_channel = type_channel.expand(-1, -1, x.shape[2], x.shape[3])            # (B, T, H, W)

        #input type channel
        x = torch.cat([x, type_channel], dim=1)

        # Down
        x1 = self.b1(x, combined_emb)
        x2 = self.pool(x1)
        x2 = self.b2(x2, combined_emb)

        # Up
        x3 = self.up1(x2)

        # Match size if needed (should be 16x16 after upsampling, but just in case)
        if x3.shape != x1.shape:
            x3 = F.interpolate(x3, size=x1.shape[2:])

        x3 = torch.cat([x3, x1], dim=1)
        x3 = self.b3(x3, combined_emb)

        return self.out(x3)
    
def add_noise(x, t, noise, alphas_cumprod):
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    
    return (
        torch.sqrt(alpha_t) * x +
        torch.sqrt(1 - alpha_t) * noise
    )

def room_to_tensor(room_matrix):
    '''
    Converts a room matrix (2D list of tile integers) into a 3D tensor with one-hot encoding across channels.
    '''
    room_matrix = np.array(room_matrix)
    height, width = room_matrix.shape

    # Initialize tensor with zeros
    tensor = np.zeros((NUM_CHANNELS, height, width), dtype=np.float32)

    for tile, channel in CHANNEL_MAP.items():
        tensor[channel] = (room_matrix == tile).astype(np.float32)

    return tensor

def compute_channel_weights(dataset):
    counts = torch.zeros(NUM_CHANNELS)

    counts = dataset.sum(dim=(0, 2, 3))  # sum over B, H, W

    freqs = counts / (counts.sum() + 1e-8)
    weights = 1.0 / (freqs + 1e-8)

    # Ensure weights are positive
    weights = torch.abs(weights)

    # normalize (optional)
    weights = weights / weights.mean()

    # clamp to prevent extreme values
    weights = torch.clamp(weights, min=0.1, max=5.0)

    # weights[0] = 1 #  wall channel
    weights[2] = 0.1 # downweight door channel

    print("Channel weights:", weights)
    return weights

def compute_type_loss(probs, room_type):
    """
    probs: (B, C, H, W) after softmax
    room_type: (B,)
    """
    B = probs.size(0)

    WALL, FLOOR, DOOR, ENEMY, CHEST, HEAL = range(6)

    loss = 0.0

    for i in range(B):
        t = room_type[i]

        enemy_amt = probs[i, ENEMY].mean()
        chest_amt = probs[i, CHEST].mean()
        heal_amt  = probs[i, HEAL].mean()

        # Adjust how strong the entities are (higher = more rigid)
        if t == "healing":
            loss += enemy_amt * 2.0
            loss += chest_amt * 1.0

        elif t == "enemy":
            loss += heal_amt * 2.0
            loss += chest_amt * 0.5

        elif t == "loot":
            loss += enemy_amt * 0.5
            loss += heal_amt * 2.0

    return loss / B

def compute_type_distributions(X, room_types, num_types):
    """
    X: (N, C, H, W)
    room_types: (N,)
    returns: (T, C)
    """
    T, C = num_types, X.size(1)     # type and tile channels
    dists = torch.zeros(T, C, device=X.device)

    for t in range(T):
        mask = (room_types == t)
        if mask.any():
            dists[t] = X[mask].mean(dim=(0,2,3))

    return dists

def create_diffusion_structure_mask(x):
    # x: (B, C, H, W)

    wall_mask = x[:, 0:1, :, :]   # WALL channel
    # door_mask = x[:, 2:3, :, :]   # DOOR channel

    # structure_mask = torch.clamp(door_mask, 0, 1)
    structure_mask = torch.zeros_like(wall_mask)

    return structure_mask  # (B, 1, H, W)

def train_diffusion_model(model, dataloader, optimizer, alphas_cumprod, epochs = 50, device='cpu'):
    model.train()

    # weights = compute_channel_weights(dataloader.dataset.X).to(device)
    dataset_dist = dataloader.dataset.X.mean(dim=(0,2,3)).to(device)  # (C,)

    type_dist = compute_type_distributions(
        dataloader.dataset.X,
        dataloader.dataset.y,
        num_types=5
    ).to(device)

    # Custom class weights for training
    class_weights = torch.tensor([
        1.0,  # wall
        1.0,  # floor
        0.3,  # door -> discourage overuse with smaller number
        1.2,  # enemy
        1.2,  # chest
        1.2   # healing
    ]).to(device)

    wall_epoch = []
    enemy_epoch = []
    chest_epoch = []
    healing_epoch = []

    for epoch in range(epochs):
        total_loss = 0

        for x, room_type in dataloader:
            x = x.to(device)                      # (B, C, H, W)
            room_type = room_type.to(device)      # (B,)
            data_dist = dataset_dist.unsqueeze(0).expand(x.size(0), -1) # expand to be same as input size 

            # Sample timestep
            t = torch.randint(0, len(alphas_cumprod), (x.size(0),), device=device)

            # Add noise
            noise = torch.randn_like(x)
            noisy_x = add_noise(x, t, noise, alphas_cumprod)

            # Predict noise
            predicted_noise = model(noisy_x, t, room_type)

            alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)

            # Reconstruct a distribution estimate
            x0_pred = (noisy_x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            probs = torch.softmax(x0_pred, dim=1)  # (B, C, H, W)

            # Compute channel distribution per batch
            pred_dist = probs.mean(dim=(2,3))  # (B, C)

            dist_loss = F.mse_loss(pred_dist, data_dist)

            # Get type loss
            type_loss = compute_type_loss(probs, room_type)

            # Get type dist loss
            target_type_dist = type_dist[room_type]  # (B, C)
            type_dist_loss = F.mse_loss(pred_dist, target_type_dist)

            # Create structure mask to zero-out loss on selected tile channels
            structure_mask = create_diffusion_structure_mask(x)
            structure_mask = structure_mask.repeat(1, x.shape[1], 1, 1) # expand to all channels just in case

            # Noise loss 
            noise_loss = ((predicted_noise - noise) ** 2) # element-wise loss
            noise_loss *= (1 - structure_mask)
            noise_loss *=  class_weights.view(1, -1, 1, 1)

            noise_loss = noise_loss.mean()

            # Wall channel consistency loss (lightly)
            wall_channel = 0

            wall_pred = probs[:, wall_channel, :, :]  # (B, H, W)
            wall_true = x[:, wall_channel, :, :]      # (B, H, W)
            wall_loss = F.mse_loss(wall_pred, wall_true)

            # Combine losses with a strength value
            loss = (
                noise_loss 
                + 0.1 * dist_loss 
                + 0.2 * wall_loss 
                + 0.2 * type_loss 
                + 0.2 * type_dist_loss
            )

            # # Encourage entities enemies, chest, healing(To Discourage flip the sign to negative)
            # x[:, 3:] -= 0.4

            # # Encourage walls
            # x[:, 0] += 0.4

            # # Encourage floor dominance
            # x[:, 1] += 0.2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            wall_epoch.append(probs[:,0].mean().item())
            enemy_epoch.append(probs[:,3].mean().item())
            chest_epoch.append(probs[:,4].mean().item())
            healing_epoch.append(probs[:,5].mean().item())

        # -----------------
        # Save model every 10 epochs
        # -----------------
        if epoch % 10 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), DIFFUSION_PATH)
            print(f"Saving Diffusion model to: {DIFFUSION_PATH}")

            # Debug: Print probability mean values of selected channels
            print("Wall prob mean:", np.mean(wall_epoch))
            print("Enemy prob mean:", np.mean(enemy_epoch))
            print("Chest prob mean:", np.mean(chest_epoch))
            print("Heal prob mean:", np.mean(healing_epoch))

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")
    
    


def generate_diffusion_dungeon_room(model, room_type, room_matrix, mask, alphas_cumprod, device):
    model.eval()
    # Convert room matrix to tensor
    room_tensor = room_to_tensor(room_matrix)       # (C, H, W)
    room_tensor = torch.tensor(room_tensor, dtype=torch.float32).unsqueeze(0).to(device)    # (1, C, H, W)
    # Convert room_tensor to hard labels
    room_labels = torch.argmax(room_tensor, dim=1, keepdim=True)  # (1,1,H,W)

    # start from noise but keep structure intact
    x = torch.randn_like(room_tensor).to(device)

    # Create one-hot version of room from noise
    room_onehot = torch.nn.functional.one_hot(room_labels.squeeze(1), num_classes=x.shape[1])
    room_onehot = room_onehot.permute(0, 3, 1, 2).float()

    # Convert mask to tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    mask_expanded = mask_tensor.repeat(1, x.shape[1], 1, 1)

    room_type_tensor = torch.tensor([room_type], device=device)

    temperature = 0.4   # How creative we want the model to be higher = chaos

    T = len(alphas_cumprod)

    tiles = torch.argmax(x, dim=1)
    unique, counts = torch.unique(tiles, return_counts=True)
    print("RAW DIFFUSION:", dict(zip(unique.tolist(), counts.tolist())))

    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.tensor([t], device=device)

            pred_noise = model(x, t_tensor, room_type_tensor)

            alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)

            # Denoising step
            x = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

            # enforce structure again by replacing those channels with original values
            # strength = 0.8
            # x = x * (1 - mask_tensor * strength) + room_tensor * (mask_tensor * strength)
            # x = torch.where(mask_expanded.bool(), room_tensor, x)
            x = x * (1 - mask_expanded) + room_onehot * mask_expanded
            

            # Add noise except final step
            if t > 0:
                noise = torch.randn_like(x)
                x += temperature * (1 - alpha_t).sqrt() * noise

    # tiles = torch.argmax(torch.softmax(x/0.5, dim=1), dim=1)
    probs = torch.softmax(x, dim=1)
    tiles = torch.multinomial(probs.permute(0,2,3,1).reshape(-1, 6), 1)
    tiles = tiles.reshape(x.shape[0], x.shape[2], x.shape[3])
    unique, counts = torch.unique(tiles, return_counts=True)
    print("POST MASK:", dict(zip(unique.tolist(), counts.tolist())))
    print("Mean probs per class:", probs.mean(dim=[0,2,3]))
    
    # Diffusion outputs continuous values, convert to discrete tile indices
    return tiles