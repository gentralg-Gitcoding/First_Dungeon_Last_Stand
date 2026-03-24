# First_Dungeon_Last_Stand
My first personal creative AI Gaming project.

Credits to PyGame for open use of base code.
https://www.pygame.org/


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
