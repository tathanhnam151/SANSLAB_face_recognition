import cv2
import time
import os
from tools.recognition import register_face

# Directory to save images
new_person_dir = "database/temp"

# Name of the person to register
person_name = "JohnTran"

# Create a directory for the person
person_dir = os.path.join(new_person_dir, person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Open the camera
cap = cv2.VideoCapture(1)

# Wait for the spacebar to be pressed
while True:
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Capture 3 images
for i in range(3):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Draw the number of taken pictures on the frame
    cv2.putText(frame, f"Picture: {i+1}", (frame.shape[1] // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Camera', frame)

    # Save the image
    cv2.imwrite(f"{person_dir}/image_{i}.jpg", frame)

    # Wait for 2 seconds
    time.sleep(2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create a new directory for the additional photos
person_dir_test = f"./database/temptest/{person_name}"
os.makedirs(person_dir_test, exist_ok=True)

# Capture frame-by-frame
ret, frame = cap.read()

# Display the resulting frame
cv2.imshow('Camera', frame)

# Save the image in the new directory
cv2.imwrite(f"{person_dir_test}/image_0.jpg", frame)

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Call the register_face function for the new photos
register_face(person_dir_test, person_name)