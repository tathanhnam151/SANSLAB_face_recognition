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


def modify_npz_only_20200424(npz_file):
    data = np.load(npz_file)
    modified_data = {}

    # Get the indices where images_name is '20200424'
    indices = data['images_name'] == '20200424'

    for key in data.files:
        values = data[key]

        # Keep only the values where images_name is '20200424'
        values = values[indices]

        modified_data[key] = values

    # Save the modified data back to the .npz file
    np.savez(npz_file, **modified_data)

def empty_npz(npz_file):
    data = np.load(npz_file)
    modified_data = {}

    for key in data.files:
        # Create an empty array with the same shape as the original data
        values = np.empty(shape=(0,) + data[key].shape[1:])

        modified_data[key] = values

    # Save the modified data back to the .npz file
    np.savez(npz_file, **modified_data)

def generate_data_npz(npz_file, num_people=1, num_photos_per_person=3):
    # Generate unique image names for each person
    image_names = np.repeat([f'0000{4500 + i}' for i in range(num_people)], num_photos_per_person)

    # Generate random embeddings for each photo
    # Assuming each embedding is a 128-dimensional vector
    image_embs = np.random.rand(num_people * num_photos_per_person, 512)

    # Save the data to the .npz file
    np.savez(npz_file, images_name=image_names, images_emb=image_embs)

def change_index(npz_file):
    data = np.load(npz_file)
    modified_data = {}

    for key in data.files:
        values = data[key]

        if key == 'images_name':
            # Find the indices where 'images_name' is '20217890'
            indices = np.where(values == '20217890')

            # Replace '20217890' with '20200424'
            values[indices] = '20200424'

        modified_data[key] = values

    # Save the modified data back to the .npz file
    np.savez(npz_file, **modified_data)

if __name__ == "__main__":
    # print_npz_contents("database/photo_datasets/face_features/feature.npz")
    # modify_npz_only_20200424("database/photo_datasets/face_features/feature.npz")
    # print_npz_contents("database/photo_datasets/face_features/feature.npz")

    # empty_npz("database/photo_datasets/face_features/feature.npz")
    # generate_data_npz("database/photo_datasets/face_features/feature.npz", num_people=1, num_photos_per_person=3)
    print_npz_contents("database/photo_datasets/face_features/feature.npz")
    print_npz_contents("downloaded_face_features.npz")