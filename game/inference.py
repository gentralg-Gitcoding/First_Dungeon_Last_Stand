import os

import torch
import numpy as np
from ai.gan_generator import Generator 
from ai.diffusion_generator import SimpleUNet, room_to_tensor
from engine.map_generator import (
    Room, clean_generated_doors, create_structure_mask, enforce_entity_limits, 
    enforce_reachable_door, enforce_room_type_bias, get_noise_schedule, remove_trapped_enemies
)
from settings import MATRIX_TO_ROOM_TILE, ROOM_HEIGHT, ROOM_WIDTH, ROOM_TYPES
import matplotlib.colors as mcolors
from huggingface_hub import hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 100  # match training
GAN_PATH = "game/data/models/generator_epoch_49.pth"
DIFFUSION_PATH = "game/data/models/diffusion_model.pth"
OG_ROOM = Room(0, 0, ROOM_WIDTH, ROOM_HEIGHT).room_map

# Tile mapping 
MATRIX_TO_TILE = {
    0: "WALL",
    1: "FLOOR",
    2: "DOOR",
    3: "ENEMY",
    4: "CHEST",
    5: "HEALING"
}

TILE_COLORS = {
    "WALL": "black",   # wall
    "FLOOR": "white",   # floor
    "DOOR": "blue",    # door
    "ENEMY": "red",     # enemy
    "CHEST": "yellow",  # chest
    "HEALING": "green"    # healing
}

ROOM_TYPES = {
    "enemy": 0,
    "loot": 1,
    "healing": 2,
    "start": 3,
    "boss": 4,
}

CMAP = mcolors.ListedColormap([
    "black",   # 0
    "white",   # 1
    "blue",    # 2
    "red",     # 3
    "yellow",  # 4
    "green"    # 5
])

class GANWrapper:
    def __init__(self, model):
        self.model = model
    
    def generate(self, room_type_str="start"):
        room = Room(0, 0, ROOM_WIDTH, ROOM_HEIGHT).room_map
        room_type = torch.tensor([ROOM_TYPES[room_type_str]], device=DEVICE)

        if room_type_str in ["start", "boss"]:
            return room

        with torch.no_grad():
            x = torch.randn(1, LATENT_DIM).to(DEVICE)
            output = self.model(x, room_type)

        return torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

class DiffusionWrapper:
    def __init__(self, model):
        self.model = model
        self.input_channels = 6
        self.num_room_types = 5

    def generate(self, room_type_str="start"):
        room = Room(0, 0, ROOM_WIDTH, ROOM_HEIGHT).room_map
        room_type = torch.tensor([ROOM_TYPES[room_type_str]], device=DEVICE)
        mask = create_structure_mask(room)
        alphas_cumprod = get_noise_schedule(device=DEVICE)

        room_tensor = room_to_tensor(room)
        room_tensor = torch.tensor(room_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # start from noise but keep structure intact
        x = torch.randn_like(room_tensor).to(DEVICE)

        # Create one-hot version of room from noise
        room_labels = torch.argmax(room_tensor, dim=1, keepdim=True)
        room_onehot = torch.nn.functional.one_hot(room_labels.squeeze(1), num_classes=self.input_channels)
        room_onehot = room_onehot.permute(0, 3, 1, 2).float()

        # Convert mask to tensor
        mask_tensor = torch.tensor(mask, dtype=torch.float32).to(DEVICE)
        mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(0).repeat(1, self.input_channels, 1, 1)

        T = len(alphas_cumprod)
        temperature = 0.4   # How creative we want the model to be higher = chaos

        with torch.no_grad():
            for t in reversed(range(T)):
                    t_tensor = torch.tensor([t], device=DEVICE)

                    pred_noise = self.model(x, t_tensor, room_type)

                    alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)

                    # Denoising step
                    x = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

                    x = x * (1 - mask_expanded) + room_onehot * mask_expanded

                    # Add noise except final step
                    if t > 0:
                        noise = torch.randn_like(x)
                        x += temperature * (1 - alpha_t).sqrt() * noise

            probs = torch.softmax(x, dim=1)
            tiles = torch.multinomial(probs.permute(0,2,3,1).reshape(-1, self.input_channels), 1)
            tiles = tiles.reshape(x.shape[0], x.shape[2], x.shape[3])

        return tiles.squeeze(0).cpu().numpy()

def load_model(model_path, model_selection):
    if model_selection == "Gans":
        model = Generator().to(DEVICE)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return GANWrapper(model)
        else:
            hf_path = hf_hub_download(
                repo_id="gentralg/GANs-Dungeon-Floor-Entities",
                filename="gans_model.pth"
            )
            print(f"Loaded model from HF repo: {hf_path}")
            model.load_state_dict(torch.load(hf_path, map_location=DEVICE))
            model.eval()
            return GANWrapper(model)
    elif model_selection == "Diffusion":
        model = SimpleUNet().to(DEVICE)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return DiffusionWrapper(model)
        else:
            hf_path = hf_hub_download(
                repo_id="gentralg/Diffusion-Dungeon-Floor-Entities",
                filename="diffusion_model.pth"
            )
            print(f"Loaded model from HF repo: {hf_path}")
            model.load_state_dict(torch.load(hf_path, map_location=DEVICE))
            model.eval()
            return DiffusionWrapper(model)

def tile_distribution(room):
    unique, counts = np.unique(room, return_counts=True)
    return {MATRIX_TO_ROOM_TILE[int(k)]: int(v) for k, v in zip(unique, counts)}     #Streamlit does not like numpy ints

def apply_original_room(room, original_room):
    '''
    Put wall edges back if generation altered it
    '''
    for y in range(len(original_room)):
        for x in range(len(original_room[0])):
            # put structural edges back
            if (
                x == 0 or x == len(original_room[0]) - 1 or
                y == 0 or y == len(original_room) - 1
            ):
                room[y][x] = original_room[y][x]
    return room

def post_process(room, room_type):
    '''Use post process functions to control and structure dungeon layout after generation'''
    room = enforce_room_type_bias(room, room_type)
    room = clean_generated_doors(room, OG_ROOM)
    room = remove_trapped_enemies(room)
    room = enforce_entity_limits(room, room_type)
    room = enforce_reachable_door(room)
    room = apply_original_room(room, OG_ROOM)
    return room
