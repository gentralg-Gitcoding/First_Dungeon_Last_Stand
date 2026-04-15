from utils.save_load_data import save_json_dataset
from utils.synthetic_data_creation import generate_training_room
from utils.data_creation_validation import is_valid_room


DATA_PATH = 'game/data/synthetic_rooms_dataset.json'


def main():
    dataset = []

    while len(dataset) < 10000:      # Enter number of rooms you want to generate

        room_matrix, room_type = generate_training_room()

        if is_valid_room(room_matrix, room_type):
            print(f'Generating Sythetic data...{len(dataset)}')
            dataset.append({f'room_matrix_{len(dataset)}': room_matrix, 'type': room_type})

    save_json_dataset(dataset, DATA_PATH)

if __name__ == "__main__":
    main()