import numpy as np

students_feature_data = np.load("database/photo_datasets/face_features/feature.npz")

def print_npz_contents(npz_file):
    data = np.load(npz_file)

    for key in data.files:
        print(f"Key: {key}")
        print(f"Value: {data[key]}")

def modify_npz_contents(npz_file):
    data = np.load(npz_file)
    modified_data = {}

    # Get the indices where images_name is '20204567'
    indices = data['images_name'] == '20204567'

    for key in data.files:
        values = data[key]

        # Keep only the values where images_name is '20204567'
        values = values[indices]

        modified_data[key] = values

    # Save the modified data back to the .npz file
    np.savez(npz_file, **modified_data)

if __name__ == "__main__":
    modify_npz_contents("database/photo_datasets/face_features/feature.npz")
    print_npz_contents("database/photo_datasets/face_features/feature.npz")