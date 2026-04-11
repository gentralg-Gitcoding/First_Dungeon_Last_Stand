from utils.synthetic_data_creation import save_dataset, generate_training_room
from utils.data_creation_validation import is_valid_room

dataset = []

while len(dataset) < 100:
    
    room_matrix, room_type = generate_training_room()

    if is_valid_room(room_matrix):
        print(f'Generating Sythetic data...{len(dataset)}')
        dataset.append({f'room_matrix_{len(dataset)}': room_matrix, 'type': room_type})

print(*dataset, sep='\n')
save_dataset(dataset)
