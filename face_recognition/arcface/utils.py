import numpy as np


def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]
    except:
        # Return arrays with the correct dimensions when no .npz file is found
        images_name = np.array([])
        images_emb = np.zeros((0, 512))

    return images_name, images_emb


# def compare_encodings(encoding, encodings):
#     sims = np.dot(encodings, encoding.T)
#     pare_index = np.argmax(sims)
#     score = sims[pare_index]
#     return score, pare_index

def compare_encodings(encoding, encodings):
    # Normalize the encodings
    encoding = encoding / np.linalg.norm(encoding)
    encodings = encodings / np.linalg.norm(encodings, axis=1, keepdims=True)
    
    # Calculate the similarity scores
    sims = np.dot(encodings, encoding.T)
    
    # Find the encoding with the highest similarity score
    pare_index = np.argmax(sims)
    score = sims[pare_index]
    
    return score, pare_index
