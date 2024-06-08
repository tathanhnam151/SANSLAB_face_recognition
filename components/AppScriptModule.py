import requests, threading, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from itertools import groupby

# Load the configuration file for the URL and other settings
# Load the configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Now you can use the settings in your script
url = config['http_url']

# def feature_backup():
#     def send_request(payload):
#         url = "https://script.google.com/macros/s/AKfycbxz9msoEMbfCL2ffGjoVAmGAXgK9FmFj067zqAoieZRT_hBEMM6luFcPLOp2XlCXYEucg/exec"
#         headers = {'Content-Type': 'application/json'}

#         # Try to send the request up to 3 times
#         for _ in range(3):
#             try:
#                 response = requests.post(url, data=json.dumps(payload), headers=headers)
#                 print("Status code:", response.status_code)

#                 # Only return if the server returns a success status code
#                 if response.status_code // 100 == 2:
#                     return response
#                 else:
#                     print(f"Server returned status code {response.status_code}, retrying...")
#             except requests.exceptions.RequestException as e:
#                 print(f"Request failed: {e}, retrying...")
            
#             time.sleep(5)  # Wait for 5 seconds before retrying

#         print("Failed to send request after 3 attempts")
#         return None

#     # Load the JSON data from a file
#     with open('database/photo_datasets/face_features/feature.json') as f:
#         data = json.load(f)

#     # Create a ThreadPoolExecutor
#     with ThreadPoolExecutor(max_workers=20) as executor:
#         futures = []
#         for item in data:
#             # Prepare the payload
#             payload = {
#                 "mssv": item['mssv'],  # replace 'mssv' with the actual key in your JSON data
#                 "face_feature": str(item['face_feature'])  # convert the face_feature array to a string
#             }

#             # Submit the task to the ThreadPoolExecutor and add the future to the list
#             futures.append(executor.submit(send_request, payload))

#             time.sleep(1)  # Wait for 1 second before sending the next request

#         # Wait for all tasks to complete and check if they were successful
#         for future in as_completed(futures):
#             result = future.result()
#             if result is None:
#                 print("A task failed")

#         # Wait for all tasks to complete
#         executor.shutdown(wait=True)

#     print("Feature backup completed")


# Feature backup

# First version sending each item as a separate request
# This function sends the face features to the server for backup, probably would be called after face registration
# def feature_backup():
#     def send_request(payload):
#         headers = {'Content-Type': 'application/json'}
#         response = requests.post(url, data=json.dumps(payload), headers=headers)
#         print("Status code:", response.status_code)
#         return response

#     # Load the JSON data from a file
#     with open('database/photo_datasets/face_features/feature.json') as f:
#         data = json.load(f)

#     for item in data:
#         # Prepare the payload
#         payload = {
#             "mssv": item['mssv'],  # replace 'mssv' with the actual key in your JSON data
#             "face_feature": str(item['face_feature'])  # convert the face_feature array to a string
#         }

#         # Send the payload and wait for the request to complete
#         send_request(payload)

#     print("Feature backup completed")

# Second version sending all items in a single request
def feature_backup():
    def send_request(payload):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print("Status code:", response.status_code)
        return response

    # Load the JSON data from a file
    with open('database/photo_datasets/face_features/feature.json') as f:
        data = json.load(f)

    # Sort data by 'mssv' and group by 'mssv'
    data.sort(key=lambda x: x['mssv'])
    groups = groupby(data, key=lambda x: x['mssv'])

    for mssv, group in groups:
        items = list(group)

        # Split items into chunks of 3
        chunks = [items[i:i + 3] for i in range(0, len(items), 3)]

        for chunk in chunks:
            # Prepare the payload
            payload = [
                {
                    "mssv": item['mssv'],
                    "face_feature": str(item['face_feature'])  # convert the face_feature array to a string
                }
                for item in chunk
            ]

            # Send the payload and wait for the request to complete
            send_request(payload)

    print("Feature backup completed")

# Mark student as attended (old version using GET request)
# def record_student_attend(mssv):
#     def send_request(mssv):
#         url_with_param = f"{url}?mssv={mssv}"
#         print(url_with_param)
#         response = requests.post(url_with_param)    
#         return response
#     threading.Thread(target=send_request, args=(mssv,)).start()
    
# Mark student as attended (new version using POST request)

# This function marks the student with the given student code as attended in Google Sheets
def record_student_attend(mssv):
    def send_request(mssv):
        payload = {'check': mssv}
        response = requests.post(url, json=payload)
        return response
    threading.Thread(target=send_request, args=(mssv,)).start()

# Get student information from Google AppScript for face registration task
# If the student hasn't been registered, new face features would be added to the database
# If the student has been registered, the face features would be updated
def get_student_info():
    # Specify the URL, file path, and file name
    url_with_param = f"{url}?getAllUser=allUser"
    print(url_with_param)
    # url_with_param = "https://script.google.com/macros/s/AKfycbycuCiYtzGzps2T2U6P9Xlj4Ns-xO4YfdFZ1MUtNe5IiqMEkf0xWTi72peq3yZf5Pk3/exec?getAllUser=allUser"
    filepath = "database/"
    filename = 'all_students.json'

    # Send a GET request to the specified URL
    response = requests.get(url_with_param)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Create directory if it doesn't exist
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Save the data into a file at the specified path
        with open(os.path.join(filepath, filename), 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        print(f"GET request to {url_with_param} failed with status code {response.status_code}.")

# Rename the file from all_students.json to students.json
# This function is for updating the student information list for face registration task
def rename_file():
    # Specify the file paths
    old_filepath = "database/students.json"
    new_filepath = "database/all_students.json"

    # Check if the old file exists and delete it
    if os.path.exists(old_filepath):
        os.remove(old_filepath)

    # Check if the new file exists and rename it
    if os.path.exists(new_filepath):
        os.rename(new_filepath, old_filepath)

# Transform the JSON format of the student information file
def transform_json_format():

    filepath = "database/students.json"
    # Load the current data
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Transform the data
    transformed_data = [
        {
            "id": item["id"],
            "name": item["name"],
            "student_code": str(item["mssv"])
        }
        for item in data
    ]

    # Write the transformed data back to the file
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)

def save_features_to_npz(features):
    # Check if features is a list
    if isinstance(features, list):
        # Separate the image names and embeddings into two different lists
        images_name = [str(feature['mssv']) for feature in features]
        images_emb = [json.loads(feature['feature']) for feature in features]
    else:
        # If features is not a list, assume it's a dictionary
        images_name = [str(features['mssv'])]
        images_emb = [json.loads(features['feature'])]

    # Convert the lists to numpy arrays
    images_name = np.array(images_name)
    images_emb = np.array(images_emb)

    # Save the numpy arrays to a .npz file with their respective keys
    np.savez_compressed('downloaded_face_features.npz', images_name=images_name, images_emb=images_emb)

def get_student_feature(feature):
    # Use the url defined at the top of the script
    global url
    file_path = f"database/photo_datasets/face_features/downloaded_feature.json"

    # Define the parameters
    params = {
        "feature": feature
    }

    # Send the GET request
    response = requests.get(url, params=params)

    # Check the status code of the response
    if response.status_code == 200:
        # If the request was successful, save the JSON data to a file
        feature_data = response.json()
        try:
            with open(file_path, 'w') as f:
                json.dump(feature_data, f)
        except Exception as e:
            print(f"Error while writing to file: {e}")  # Print any exceptions while writing to file

        # Save the features to a .npz file
        save_features_to_npz(feature_data)

        return feature_data
    else:
        # If the request was not successful, print an error message and return None
        print(f"Error: Received status code {response.status_code}")
        return None
    
def format_json_file(file_path):
    # Read the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Write the JSON data back to the file with indentation
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def check_for_updates():
    def update():
        while True:
            print("\n")  # Print a blank line
            print("Checking for database update ...")
            print("\n")  # Print another blank line
            time.sleep(5)  # Wait for 5 seconds

    # Create a new thread that will run the update function
    update_thread = threading.Thread(target=update)

    # Make the thread a daemon thread
    update_thread.daemon = True

    # Start the new thread
    update_thread.start()





