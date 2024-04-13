import json
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QCompleter,  QCompleter, QComboBox

students_list = "database/students.json"

from components.FaceRegistrationWindow import FaceRegistrationWindow   

class FaceRegistrationDialog(QDialog):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Face Registration")
        
        self.mssv_label = QLabel("MSSV:")
        self.mssv_input = QComboBox()
        self.mssv_input.setEditable(True)  # Make the QComboBox editable

        self.name_label = QLabel("Name:")
        self.name_display = QLabel("")  # Add a QLabel to display the student's name
        
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self.proceed_registration)
        self.proceed_button.setEnabled(False)  # Disable the button initially
        
        layout = QVBoxLayout()
        layout.addWidget(self.mssv_label)
        layout.addWidget(self.mssv_input)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_display)
        layout.addWidget(self.proceed_button)
        
        self.setLayout(layout)  # Set the layout directly on the QDialog
        
        self.load_students()
        
    def load_students(self):
        # Load students from JSON file
        with open(students_list) as file:
            students = json.load(file)
        
        # Extract student codes
        student_codes = [student['student_code'] for student in students]

        # Add an empty string at the beginning 
        student_codes.insert(0, "")

        # Populate the QComboBox and set up QCompleter
        self.mssv_input.addItems(student_codes)
        completer_codes = QCompleter(student_codes, self)
        self.mssv_input.setCompleter(completer_codes)
        self.mssv_input.currentTextChanged.connect(self.validate_input)  # Connect the currentTextChanged signal to the validate_input method
        
    def validate_input(self, text):
        # Enable the button if the input text is a valid student code
        with open(students_list) as file:
            students = json.load(file)
        student_codes = [student['student_code'] for student in students]
        valid_input = self.mssv_input.currentText() in student_codes and self.mssv_input.currentText() != ""
        self.proceed_button.setEnabled(valid_input)

        # Update the name display
        if valid_input:
            for student in students:
                if student['student_code'] == self.mssv_input.currentText():
                    self.name_display.setText(f'Name: {student["name"]}')
                    break
        else:
            self.name_display.setText("")

    def proceed_registration(self):
        # Read the JSON file
        with open(students_list, 'r') as f:
            students = json.load(f)

        # The student code you're looking for
        student_code = self.mssv_input.currentText()

        # Create and show the FaceRegistrationWindow
        self.face_registration_window = FaceRegistrationWindow(student_code)
        self.face_registration_window.closed.connect(self.close)
        self.face_registration_window.show()

    def closeEvent(self, event) -> None:
        self.done(0)
        return super().closeEvent(event)