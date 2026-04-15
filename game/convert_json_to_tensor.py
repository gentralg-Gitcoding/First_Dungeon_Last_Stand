from utils.save_load_data import load_json_dataset, save_tensor_dataset
from utils.data_to_tensors_converter import dataset_to_tensors

DATA_SYNTH_PATH = 'game/data/synthetic_rooms_dataset.json'
DATA_TENSOR_PATH = 'game/data/synthetic_rooms_tensors.npz'


def main():
    dataset = load_json_dataset(DATA_SYNTH_PATH)

    tensors, labels = dataset_to_tensors(dataset)

    save_tensor_dataset(tensors, labels, DATA_TENSOR_PATH)

if __name__ == "__main__":
    main()