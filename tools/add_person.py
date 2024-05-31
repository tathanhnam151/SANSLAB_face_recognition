import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the face detector 
detector = SCRFD(model_file="./face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Face recognizer
# recognizer = iresnet_inference(
#     model_name="r34", path="./face_recognition/arcface/weights/arcface_r34.pth", device=device
# )

recognizer = iresnet_inference(
    model_name="r34", path="./face_recognition/arcface/weights/arcface_r34.pth", device=device
)

NEW_PERSON_DIR = "./database/temp/"


@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Define a series of image preprocessing steps
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert the image to RGB format
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Apply the defined preprocessing to the image
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Use the model to obtain facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    """
    Add a new person to the face recognition database.

    Args:
        backup_dir (str): Directory to save backup data.
        add_persons_dir (str): Directory containing images of the new person.
        faces_save_dir (str): Directory to save the extracted faces.
        features_path (str): Path to save face features.
    """
    # Initialize lists to store names and features of added images
    images_name = []
    images_emb = []

    # Read the folder with images of the new person, extract faces, and save them
    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)

        # Check if the path is a directory before proceeding
        if not os.path.isdir(person_image_path):
            continue

        # Create a directory to save the faces of the person
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                # Detect faces and landmarks using the face detector
                bboxes, landmarks = detector.detect(image=input_image)

                # Extract faces
                for i in range(len(bboxes)):
                    # Get the number of files in the person's path
                    number_files = len(os.listdir(person_face_path))

                    # Get the location of the face
                    x1, y1, x2, y2, score = bboxes[i]

                    # Extract the face from the image
                    face_image = input_image[y1:y2, x1:x2]

                    # Path to save the face
                    path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                    # Save the face to the database
                    cv2.imwrite(path_save_face, face_image)

                    # Extract features from the face
                    images_emb.append(get_feature(face_image=face_image))
                    images_name.append(name_person)

    # Check if no new person is found
    if images_emb == [] and images_name == []:
        print("No new person found!")
        return None

    # Convert lists to arrays
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    # Read existing features if available
    features = read_features(features_path)

    if features is not None:
        # Unpack existing features
        old_images_name, old_images_emb = features

        # Combine new features with existing features
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))

        print("Updated features!")

    # Save the combined features
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Move the data of the new person to the backup data directory
    for sub_dir in os.listdir(add_persons_dir):
        dir_to_move = os.path.join(add_persons_dir, sub_dir)
        dir_to_backup = os.path.join(backup_dir, sub_dir)

        # Check if the directory already exists in the backup directory
        if os.path.exists(dir_to_backup):
            # If it does, check if it's a directory
            if os.path.isdir(dir_to_backup):
                # If it's a directory, remove it
                shutil.rmtree(dir_to_backup)
            else:
                # If it's not a directory, remove the file
                os.remove(dir_to_backup)

        shutil.move(dir_to_move, backup_dir, copy_function=shutil.copytree)
        
    print("Successfully added new person!")

def add_person_for_verification():
    """Add a new person for verification."""
    images_name = []
    images_emb = []
    images_landmark = []

    if not os.path.exists(NEW_PERSON_DIR):
        print(f"Directory not found: {NEW_PERSON_DIR}")
        return None, None, None

    for person_dir in os.listdir(NEW_PERSON_DIR):
        person_dir_path = os.path.join(NEW_PERSON_DIR, person_dir)
        if os.path.isdir(person_dir_path):
            for img_name in os.listdir(person_dir_path):
                img_path = os.path.join(person_dir_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                print(f"Successfully read image: {img_path}")

                # Detect faces and landmarks
                bboxes, landmarks = detector.detect(img)

                if len(bboxes) == 0:
                    print(f"No faces found in image: {img_path}")
                    continue

                print(f"Faces found in image: {img_path}")

                for i, bbox in enumerate(bboxes):
                    # Ignore confidence score and always extract face and features
                    x1, y1, x2, y2, _ = bbox
                    face = img[y1:y2, x1:x2]
                    print(f"Face extracted from image: {img_path}")

                    emb = get_feature(face)
                    print(f"Features extracted from face in image: {img_path}")

                    images_name.append(person_dir)
                    images_emb.append(emb)
                    images_landmark.append(landmarks[i])

    if len(images_emb) == 0 or len(images_name) == 0:
        print("No new person found!")
        return None, None, None

    images_name = np.array(images_name)
    images_emb = np.array(images_emb)
    images_landmark = np.array(images_landmark)

    print(f"images_name: {images_name}")
    print(f"images_emb: {images_emb}")
    print(f"images_landmark: {images_landmark}")

    return images_name, images_emb, images_landmark

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="database/photo_datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="database/photo_datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="database/photo_datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="database/photo_datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_persons(**vars(opt))
