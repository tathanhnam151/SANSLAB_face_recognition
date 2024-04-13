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


def compare_encodings(encoding, encodings):
    sims = np.dot(encodings, encoding.T)
    pare_index = np.argmax(sims)
    score = sims[pare_index]
    return score, pare_index
