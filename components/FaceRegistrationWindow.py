from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QWidget, QMessageBox

from components.CameraModule import Camera
import os, cv2, shutil, time
from tools.recognition import register_face
from tools.add_person import add_persons

class FaceRegistrationWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self, student_code):
        super().__init__()
        self.setWindowTitle("Face Registration")

        self.camera = Camera()
        self.student_code = student_code

        self.camera_label = QLabel()
        self.take_picture_button = QPushButton("Take Picture")
        self.take_picture_button.clicked.connect(self.take_picture)

        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.take_picture_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update the frame every 30 ms

        self.camera.start()

        # Directory to save images
        self.new_person_dir = "database/temp"

        # Create a directory for the person
        self.person_dir = os.path.join(self.new_person_dir, self.student_code)
        if not os.path.exists(self.person_dir):
            os.makedirs(self.person_dir)

        self.image_counter = 0

    def update_frame(self):
        ret, frame = self.camera.get_frame()
        if ret:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)

    def take_picture(self):
        print("Picture taken")

        # Capture frame-by-frame
        ret, frame = self.camera.get_frame()

        # Convert the frame back to BGR before saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save the image
        cv2.imwrite(f"{self.person_dir}/image_{self.image_counter}.jpg", frame_bgr)

        self.image_counter += 1

        if self.image_counter == 3:
            self.take_picture_button.setText("Finish Registration")
            self.take_picture_button.clicked.disconnect()
            self.take_picture_button.clicked.connect(self.handleRegistration)

            # Create a new directory for the additional photos
            person_dir_test = f"./database/temptest/{self.student_code}"
            os.makedirs(person_dir_test, exist_ok=True)

            # Save the image in the new directory
            cv2.imwrite(f"{person_dir_test}/image_0.jpg", frame_bgr)

    def handleRegistration(self):
        # Call the register_face function for the new photos
        registration_result = register_face(self.person_dir, self.student_code)

        if registration_result:
            QMessageBox.information(self, "Registration", "Registration successful!")

            # Copy the entire directory to the new directory
            new_persons_to_be_added = "database/photo_datasets/new_persons"
            os.makedirs(new_persons_to_be_added, exist_ok=True)

            shutil.copytree(self.person_dir, f"{new_persons_to_be_added}/{self.student_code}")
            backup_dir = "database/photo_datasets/backup"
            add_persons_dir = "database/photo_datasets/new_persons"
            faces_save_dir = "database/photo_datasets/data/"
            features_path = "database/photo_datasets/face_features/feature"
            add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path)

        else:
            QMessageBox.warning(self, "Registration", "Registration failed. Please try again.")

        # Add a delay before deleting the directories
        time.sleep(1)

        # Empty the directories no matter the result
        shutil.rmtree(self.new_person_dir, ignore_errors=True)
        shutil.rmtree(f"./database/temptest/", ignore_errors=True)

        # Recreate the directories
        os.makedirs(self.new_person_dir, exist_ok=True)
        os.makedirs(f"./database/temptest/", exist_ok=True)

        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.camera.stop()
        self.closed.emit()
        super().closeEvent(event)