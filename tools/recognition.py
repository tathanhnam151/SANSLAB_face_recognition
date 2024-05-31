import cv2, os, torch, time
from torchvision import transforms

import numpy as np

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from tools.jsonconverter import get_student_info

from tools.add_person import add_person_for_verification

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector
detector = SCRFD(model_file="./face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Face recognizer

recognizer = iresnet_inference(
    model_name="r34", path="./face_recognition/arcface/weights/arcface_r34.pth", device=device
)

# recognizer = iresnet_inference(
#     model_name="r18", path="./face_recognition/arcface/weights/arcface_r18.pth", device=device
# )

# # Load precomputed face features and names
# images_names, images_embs = read_features(feature_path="database/photo_datasets/face_features/test_feature")

@torch.no_grad()
def get_feature(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Inference to get feature
    emb_img_face = recognizer(face_image).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb


def recognition(face_image):
    """
    Recognize a face image.

    Args:
        face_image: The input face image.

    Returns:
        tuple: A tuple containing the recognition score and name.
    """

    # Reread precomputed face features and names
    images_names, images_embs = read_features(feature_path="database/photo_datasets/face_features/feature")
    
    # Check if face_image is not empty
    if face_image.size == 0:
        return None, None

    # Get feature from face
    query_emb = get_feature(face_image)

    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]

    return score, name

import faiss

def recognition_faiss(face_image):
    """
    Recognize a face image.

    Args:
        face_image: The input face image.

    Returns:
        tuple: A tuple containing the recognition score and name.
    """

    # Reread precomputed face features and names
    images_names, images_embs = read_features(feature_path="database/photo_datasets/face_features/feature")

    # Check if face_image is not empty
    if face_image.size == 0:
        return None, None

    # Get feature from face
    query_emb = get_feature(face_image).reshape(1, -1)

    # Check if the dimensions match
    assert query_emb.shape[1] == images_embs.shape[1], "Dimension mismatch"


    # Convert to float32
    query_emb = query_emb.astype('float32')
    images_embs = images_embs.astype('float32')

    # Normalize the embeddings to have L2 norm = 1
    faiss.normalize_L2(query_emb)
    faiss.normalize_L2(images_embs)

    # Build the FAISS index
    index = faiss.IndexFlatL2(images_embs.shape[1])
    index.add(images_embs)

    # Perform the search
    D, I = index.search(query_emb, 1)

    # Get the ID of the closest image
    id_min = I[0][0]

    # Get the name of the closest image
    name = images_names[id_min]

    # Get the score of the closest image
    score = D[0][0]

    # Convert the score to cosine similarity
    cosine_similarity = 1 - score / 2

    return cosine_similarity, name

def start_face_detection_and_recognition():
    """Start face detection and recognition."""
    cap = cv2.VideoCapture(1)

    # Initialize the FPS counter
    fps_counter = 0
    start_time = time.time()
  
    while True:
        _, img = cap.read()

        # Face detection
        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=img)

        if outputs is not None:
            for i in range(len(bboxes)):
                face_image = img[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
                face_align = norm_crop(img, landmarks[i])  
                score, student_code = recognition(face_image=face_align)
                if student_code is not None:
                    # Convert the student code to the student's name
                    student_info = get_student_info(json_file = "database/students.json", search_param=student_code)
                    if student_info:
                        name, _ = student_info
                    else:
                        name = "UNKNOWN"

                    if score < 0.40:
                        caption = "UNKNOWN"
                    else:
                        caption = f"{name}:{score:.2f}"

                    cv2.rectangle(img, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), (0, 255, 0), 2)
                    cv2.putText(img, caption, (bboxes[i][0], bboxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                # Display the aligned face in a separate window
                cv2.imshow("Aligned Face", face_align)

        # Calculate and display FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        fps = fps_counter / elapsed_time
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow("Face Recognition", img)

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def register_face(new_person_dir, person_name):
    """
    Register a new face in the face recognition database.
    """

    # Add the new person for verification
    images_name, images_emb, images_landmark = add_person_for_verification()

    # If no new person was found, return
    if images_name is None:
        return False

    # Initialize the score counter
    score_counter = 0

    # Get list of images in the directory
    image_files = os.listdir(new_person_dir)

    # Read the first image
    frame = cv2.imread(os.path.join(new_person_dir, image_files[0]))

    # Detect faces and landmarks
    bboxes, landmarks = detector.detect(frame)

    for i, bbox in enumerate(bboxes):
        # Extract face
        x1, y1, x2, y2, _ = bbox
        face = frame[y1:y2, x1:x2]

        # Get feature from face
        query_emb = get_feature(face)

        # Compare with stored encodings
        score, id_min = compare_encodings(query_emb, images_emb)
        print("Score:", score)
        name = images_name[id_min]
        score = score[0]

        # If the score is above 0.4 and the name matches, increment the score counter
        if score > 0.4 and name == person_name:
            score_counter += 1

    # If the score counter was above 0.4, the registration was successful
    if score_counter >= 1:
        print("Registration successful!")
        return True
    else:
        print("Registration failed. Please try again.")
        print("Score counter:", score_counter)
        return False

if __name__ == "__main__":
    start_face_detection_and_recognition()