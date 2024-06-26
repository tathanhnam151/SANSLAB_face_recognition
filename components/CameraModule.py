import datetime, time, os, json, cv2

import numpy as np

from tools.jsonconverter import get_student_info
from tools.recognition import recognition, recognition_faiss, detector
from face_alignment.alignment import norm_crop
from PIL import ImageFont, ImageDraw, Image


rtsp_url = "rtsp://admin:sanslab1@192.168.1.64:554/Streaming/Channels/101"

class Camera:
    def __init__(self, config_file='config.json'):
        self.camera = None
        self.recognition_start_times = {}
        self.attendance = []

        # Load the configuration file
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def start(self):
        camera_option = self.config.get('camera-option', 0)
        if isinstance(camera_option, int):
            # Use the specified camera index
            self.camera = cv2.VideoCapture(camera_option)
        elif isinstance(camera_option, str):
            # Use the specified RTSP URL
            self.camera = cv2.VideoCapture(camera_option)
        else:
            raise ValueError(f"Invalid camera option: {camera_option}")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def get_frame(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))  # Resize the frame
                return True, frame
        return False, None

    def get_frame_recognition(self):
        start_time = time.time()

        if self.camera is not None:
            ret, img = self.camera.read()
            if ret:
                # Face detection
                outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=img)

                recognition_results = []
                recognized_names = []
                if outputs is not None:
                    for i in range(len(bboxes)):
                        face_image = img[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
                        face_align = norm_crop(img, landmarks[i])  
                        # score, student_code = recognition(face_image=face_align)
                        score, student_code = recognition_faiss(face_image=face_align)
                        print(score, student_code)
                        if student_code is not None and score >= 0.40:
                            # Convert the student code to the student's name
                            student_info = get_student_info(json_file = "database/students.json", search_param=student_code)
                            if student_info:
                                name, _ = student_info
                                recognized_names.append(student_code)
                            else:
                                name = "UNKNOWN"
                            caption = f"{name}:{score:.2f}"
                        else:
                            caption = "Unknown"
                            name = "Unknown"

                        # Convert the image to a PIL Image
                        img_pil = Image.fromarray(img)
                        draw = ImageDraw.Draw(img_pil)

                        # Specify the font and size
                        font = ImageFont.truetype('Arial Unicode.ttf', 30)

                        # Draw the text on the image
                        draw.text((bboxes[i][0], bboxes[i][1] - 10), caption, font=font, fill=(36,255,12,0))

                        # Convert the PIL Image back to an OpenCV image
                        img = np.array(img_pil)

                        # Draw the bounding box on the image
                        cv2.rectangle(img, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), (0, 255, 0), 2)

                        recognition_results.append((score, name))

                    # Check for attendance
                    self.check_for_attendance(recognized_names)

                    # Calculate FPS
                    fps = 1.0 / (time.time() - start_time)
                    fps_text = f"FPS: {fps:.2f}"

                    # Draw FPS on the image
                    cv2.putText(img, fps_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)                

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640, 480))  # Resize the image
                    return True, img, recognition_results

        return False, None, []

    def check_for_attendance(self, recognized_names):
        for name in list(self.recognition_start_times.keys()):
            if name not in recognized_names:
                del self.recognition_start_times[name]
            elif datetime.datetime.now() - self.recognition_start_times[name] >= datetime.timedelta(seconds=2):
                self.attendance.append((name, datetime.datetime.now()))
                print(f"{name} is present.")
                del self.recognition_start_times[name]
        for name in recognized_names:
            if name not in self.recognition_start_times:
                self.recognition_start_times[name] = datetime.datetime.now()

    def take_photos(self, new_person_dir, person_name):
        countdowns = [3, 2, 2]  # Countdown times before each photo
        for i in range(3):
            countdown = countdowns[i]
            while countdown >= 0:
                ret, frame = self.get_frame()
                if not ret:
                    print("Failed to get frame")
                    return

                # Display the countdown on the frame
                cv2.putText(frame, str(countdown), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the frame
                cv2.imshow('Camera', frame)

                # Wait for 1 second
                cv2.waitKey(1000)

                countdown -= 1

            # Save the frame as an image
            person_dir = os.path.join(new_person_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            cv2.imwrite(os.path.join(person_dir, f'photo{i+1}.jpg'), frame)

            # Wait for 2 seconds before the next photo
            time.sleep(2)

        cv2.destroyAllWindows()