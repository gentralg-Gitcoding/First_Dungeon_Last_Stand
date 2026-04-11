# First_Dungeon_Last_Stand
My first personal creative AI Gaming project.

Credits to PyGame for open use of base code.
https://www.pygame.org/

Credits for artwork used 
rougelike tiles - https://opengameart.org/content/roguelike-tiles-large-collection
npc sprites - https://opengameart.org/content/700-sprites
Floor art credit - https://opengameart.org/content/tileable-brick-ground-textures-set-2
wall art credit - https://opengameart.org/content/stones-brick-textures
door art - https://opengameart.org/content/lpc-windows-doors

How we want to setup this project:
game/
│
├── main.py
├── settings.py
│
├── engine/
│   ├── map_generator.py
│   ├── entity_system.py
│   ├── combat.py
│   └── inventory.py
│
├── assets/
│   ├── tiles/
│   ├── player/
│   ├── enemies/
│   └── items/
│
├── data/
│   ├── enemy_stats.json
│   ├── item_stats.json
│   └── dungeon_rules.json
│
└── ai/
    ├── enemy_ai.py
    └── procedural_generation.py
