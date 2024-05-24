import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime

from components.AppScriptModule import *
from components.CameraModule import Camera
from components.FaceRegistrationDialog import FaceRegistrationDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition")
        
        self.camera = Camera()
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        self.face_registration_button = QPushButton("Face Registration")
        self.face_registration_button.clicked.connect(self.show_face_registration_dialog) 
        
        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.face_registration_button)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

        # Start the camera
        self.camera.start()

        # Initialize the message box
        self.msg_box = None

        # Perform feature backup
        # feature_backup()

        # Get all student features from Google AppScript
        get_student_feature("20200424")
        format_json_file("database/photo_datasets/face_features/downloaded_feature.json")
        
        # Get all student information and store in a dictionary
        get_student_info()
        # Copy student information to the original file
        rename_file()
        transform_json_format()


    def update_camera_frame(self):
        ret, frame, recognized_faces = self.camera.get_frame_recognition()
        if ret:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)

            # Check attendance
            for (name, time) in list(self.camera.attendance):  # Make a copy of the list for iteration
                # Record attendance for the student
                print(name, time)
                record_student_attend(name)

                if self.msg_box is None:
                    self.msg_box = NonBlockingMessageBox("Attendance", f"{name} is present at {time}", self)
                    self.msg_box.finished.connect(self.on_msg_box_finished)  # Connect the finished signal
                    self.msg_box.move(self.pos().x(), self.pos().y())
                else:
                    self.msg_box.setText(self.msg_box.text() + f"\n{name} is present at {time}")
                    self.msg_box.reset_timer()  # Reset the timer of the message box
                self.camera.attendance.remove((name, time))  # Remove the attendance record

        # Remove the attendance record of a person from the message box after 5 seconds
        if self.msg_box is not None:
            lines = self.msg_box.text().split("\n")
            if len(lines) > 1:
                record_time = datetime.strptime(lines[0].split(" ")[-1], "%H:%M:%S.%f")
                if (datetime.now() - record_time).total_seconds() >= 5:
                    self.msg_box.setText("\n".join(lines[1:]))

    def on_msg_box_finished(self):
        self.msg_box = None  # Set self.msg_box to None when the message box is closed

    def show_face_registration_dialog(self):
        #Stop the camera
        self.camera.stop()
        self.camera_label.setText("Camera is being used for face registration. Please wait...")

        # Show the FaceRegistrationDialog
        self.face_registration_dialog = FaceRegistrationDialog(self)
        self.face_registration_dialog.finished.connect(self.on_dialog_finished)
        self.face_registration_dialog.show()

    def on_dialog_finished(self):
        # Start the camera and update the label text
        self.camera.start()
        print("Camera is ready for use.")

class NonBlockingMessageBox(QMessageBox):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(text)
        self.setStandardButtons(QMessageBox.Ok)
        self.buttonClicked.connect(self.close)
        self.setWindowModality(Qt.NonModal)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)
        self.timer.start(5000)  # 5 seconds
        self.show()

    def reset_timer(self):
        self.timer.start(5000)  # Reset the timer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())