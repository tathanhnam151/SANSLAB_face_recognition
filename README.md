# SANSLAB_face_recognition

# Attendance System

This project is an advanced Attendance System that leverages facial recognition technology and provides a user-friendly interface. The system uses SCRFD for face detection, ArcFace for face recognition, and FAISS for efficient similarity search. The front-end is built with PyQt to provide an interactive user interface, and the system integrates with Google Sheets via AppScript to allow users to manage attendance data seamlessly.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Configuration](#configuration)
- [Contribution](#contribution)
- [License](#license)

## Features

- **Face Detection**: Utilizes SCRFD for efficient and accurate face detection.
- **Face Recognition**: Employs ArcFace for robust face recognition.
- **Similarity Search**: Uses FAISS for fast and scalable similarity search.
- **User Interface**: Built with PyQt for a responsive and intuitive user interface.
- **Google Sheets Integration**: Allows interaction with Google Sheets for attendance data management.
- **Apps Script Integration**: Facilitates advanced operations through Google Apps Script.

## Installation

To get started with the Attendance System, follow these steps:

1. **Clone the repository**:
  ```bash
    git clone https://github.com/tathanhnam151/SANSLAB_face_recognition
    cd attendance-system
  ```

2. **Install miniconda**:

- Download and install miniconda from [here](https://docs.anaconda.com/free/miniconda/index.html)

3. **Set up a Conda environment**:
  ```bash
    conda create -n face-dev9 python=3.9
    conda activate face-dev9
  ```

4. **Install dependencies**:
  ```bash
    pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
  ```
5. **Install Model Weights**:

- Check the README.md files inside the face_detection/scrfd/weights and face_recognition/arcface/weights directories for instructions on installing the pretrained weights.
- The app runs best when using scrfd_2.5g_bnkps for face detection and arcface_r34 for face recognition.

## Usage

To run the Attendance System:

1. **Start the application**:

  ```bash
    python app.py
  ```

2. **User Interface**:

- The main window will open, allowing you to interact with the system.
- Mark attendance by recognizing registered users.
- Register new users by capturing their facial images.

3. **Google Sheets Integration**:

- Attendance data and user profiles for facial recognition can be synced with a Google Sheets document, providing a convenient backup and enabling quick setup on new devices.
- Users can view, edit, and update attendance records and student information in real-time, facilitating seamless management and collaboration.

## Technologies

- SCRFD: Used for face detection. [GitHub](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- ArcFace: Employed for face recognition. [GitHub](https://github.com/deepinsight/insightface)
- FAISS: Utilized for similarity search. [GitHub](https://github.com/facebookresearch/faiss)
- PyQt: Used for building the user interface. [PyQt Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- AppScript: For advanced interaction with Google Sheets. [Google Apps Script](https://www.google.com/script/start/)
## Configuration

Configuration settings can be adjusted in the config.py file (TODO).
Ensure correct paths for model files and other resources.

## Contribution

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details (TODO).
