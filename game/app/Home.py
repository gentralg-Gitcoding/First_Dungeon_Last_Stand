import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from inference import CMAP, DIFFUSION_PATH, GAN_PATH, OG_ROOM, ROOM_TYPES, TILE_COLORS, load_model, post_process, tile_distribution

st.set_page_config(page_title="Dungeon Generator", layout="centered")

st.title("🧱 Dungeon Room Generator")

@st.cache_resource
def get_model(model_selection):
    if model_selection == "Gans":
        return load_model(GAN_PATH, model_selection)
    elif model_selection == "Diffusion":
        return load_model(DIFFUSION_PATH, model_selection)

model_selection = st.selectbox("Model:", options=["Gans", "Diffusion"])

model = get_model(model_selection)

room_type_selection = st.selectbox("Optionally - Select room type:", options=ROOM_TYPES)  # ["enemy", "loot", "healing", "start", "boss"]

if "room" not in st.session_state:
    st.session_state.room = OG_ROOM
    st.session_state.post_room = OG_ROOM

if st.button("Generate Room"):
    if model_selection == "Gans":
        st.session_state.room = model.generate(room_type_selection)
        st.session_state.post_room = post_process(model.generate(room_type_selection), room_type_selection)
    else:
        st.session_state.room = model.generate(room_type_selection)
        st.session_state.post_room = post_process(model.generate(room_type_selection), room_type_selection)

    st.subheader(f"Generated {model_selection} Room Grid")
room = st.session_state.room
post_room = st.session_state.post_room

fig, ax = plt.subplots(1, 2, figsize=(18, 10))
ax[0].imshow(room, cmap=CMAP, vmin=0, vmax=5)
ax[0].set_title("Tile Map")
ax[0].axis("off")

ax[1].imshow(post_room, cmap=CMAP, vmin=0, vmax=5)
ax[1].set_title("Tile Map")
ax[1].axis("off")

legend_patches = [mpatches.Patch(color=color, label=tile) for tile, color in TILE_COLORS.items()]
ax[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Tile Distribution")
    st.write(tile_distribution(room))
with col2:
    st.subheader("Tile Dist Post Process")
    st.write(tile_distribution(post_room))





