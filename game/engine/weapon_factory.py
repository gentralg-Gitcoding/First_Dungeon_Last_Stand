import pygame

from utils import save_load_data

WEAPON_DATASET = save_load_data.load_json_dataset("game/data/weapon_stats.json")

def create_weapon(weapon_id):
    data = WEAPON_DATASET[weapon_id]

    if data["category"] == "sword":
        return Sword(data)


class Weapon():
    def __init__(self, data):
        self.category = data["category"]
        self.name = data["name"]
        self.damage = data["damage"]
        self.attack_speed = data["attack_speed"]
        self.range = data["range"]
        self.last_attack_time = 0
        self.attacking = False

    def update():
        pass

    def attack(self):
        current_time = pygame.time.get_ticks()

        if current_time - self.last_attack_time >= self.attack_speed:
            self.attacking = True
            self.last_attack_time = current_time
            return self.damage

class MeleeWeapon(Weapon):
    def __init__(self, data):
        super().__init__(data)

        self.max_angle = data["max_angle"]
        self.attack_angle = 0

    def update(self):
        current_time = pygame.time.get_ticks()

        # ------------------------
        # Attacking Logic for weapons
        # ------------------------
        if self.attacking:
            self.attack_angle += 10

            if self.attack_angle >= self.max_angle or current_time - self.last_attack_time >= self.attack_speed:
                self.attacking = False
                self.attack_angle = 0
                self.last_attack_time = current_time



class Sword(MeleeWeapon):
    def __init__(self, data):
        super().__init__(data)



