import numpy as np
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def npz_to_json(npz_file, json_file):
    # Load your npz file
    data = np.load(npz_file)

    # Get the image names and face features
    image_names = data['images_name']
    face_features = data['images_emb']

    # Initialize an empty list to hold the data
    data_list = []

    # Iterate over the image names and face features
    for i in range(len(image_names)):
        # Create a dictionary for each image
        image_dict = {
            'mssv': image_names[i],
            'face_feature': face_features[i]
        }

        # Add the dictionary to the list
        data_list.append(image_dict)

    # Convert the list to a JSON string with indentation
    encoded_numpy_data = json.dumps(data_list, cls=NumpyArrayEncoder, indent=4)

    # Write the JSON string to a file
    with open(json_file, 'w') as f:
        f.write(encoded_numpy_data)


def npz_to_json_no_gap(npz_file, json_file):
    # Load your npz file
    data = np.load(npz_file)

    # Get the image names and face features
    image_names = data['images_name']
    face_features = data['images_emb']

    # Initialize an empty list to hold the data
    data_list = []

    # Iterate over the image names and face features
    for i in range(len(image_names)):
        # Create a dictionary for each image
        image_dict = {
            'mssv': image_names[i],
            'face_feature': face_features[i]
        }

        # Add the dictionary to the list
        data_list.append(image_dict)

    # Convert the list to a JSON string without indentation
    encoded_numpy_data = json.dumps(data_list, cls=NumpyArrayEncoder)

    # Write the JSON string to a file
    with open(json_file, 'w') as f:
        f.write(encoded_numpy_data)

if __name__ == "__main__":
    npz_file = 'database/photo_datasets/face_features/feature.npz'  # replace with your npz file path
    json_file = 'database/photo_datasets/face_features/feature.json'  # replace with your json file path

    npz_to_json_no_gap(npz_file, json_file)