import numpy as np
import pandas as pd

def npz_to_csv(npz_file, csv_file):
    data = np.load(npz_file)

    # Prepare a dictionary to hold the data
    data_dict = {}

    for key in data.files:
        values = data[key]

        if values.ndim > 1:
            # If values is a 2D array, store each element as a separate item
            for i in range(values.shape[1]):
                data_dict[f'{key}_{i}'] = values[:, i]
        else:
            data_dict[key] = values

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_dict)

    # Save the DataFrame to a .csv file
    df.to_csv(csv_file, index=False)

def csv_to_npz(csv_file, npz_file):
    # Load the data from the .csv file
    df = pd.read_csv(csv_file)

    # Prepare a dictionary to hold the data to be saved to the .npz file
    data = {}

    for column in df.columns:
        if '_' in column:
            # This column was originally an element of a 2D array
            key, index = column.rsplit('_', 1)

            if not index.isdigit():
                continue

            index = int(index)

            index = int(index)

            if key not in data:
                # Initialize an empty list for this key
                data[key] = []

            # Append the data for this column to the list for this key
            data[key].append(df[column].values)
        else:
            # This column was originally a 1D array
            data[column] = df[column].values

    # Convert the lists for the keys that were originally 2D arrays back to 2D arrays
    for key in data:
        if isinstance(data[key], list):
            data[key] = np.array(data[key]).T

    # Save the data to a .npz file
    np.savez(npz_file, **data)

if __name__ == "__main__":
    npz_file = 'database/photo_datasets/face_features/feature.npz'  # Replace with your .npz file
    csv_file = 'database/photo_datasets/face_features/feature.csv'  # Replace with your .csv file
    test_npz_file = 'database/photo_datasets/face_features/test_feature.npz'

    # npz_to_csv(npz_file, csv_file)
    # csv_to_npz(csv_file, test_npz_file)
    df = pd.read_csv(csv_file)
    print(df.dtypes)