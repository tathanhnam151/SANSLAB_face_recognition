import time

import cv2

from face_detection.scrfd.detector import SCRFD

# Initialize the face detector
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")


def main():
    
    # # RTSP URL of your camera feed
    # http_url = "https://bitdash-a.akamaihd.net/content/MI201109210084_1/m3u8s-fmp4/f08e80da-bf1d-4e3d-8899-f0f6155f6efa.m3u8"

    # # Open the RTSP stream
    # cap = cv2.VideoCapture(http_url)

    # # Check if the camera is opened successfully
    # if not cap.isOpened():
    #     print("Error: Could not open camera.")
    #     exit()
    
    # Open the camera
    cap = cv2.VideoCapture(1)

    # Initialize variables for measuring frame rate
    start = time.time_ns()
    frame_count = 0
    fps = -1

    # Save video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(
        "results/face-detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size
    )

    # Read frames from the camera
    while True:
        # Capture a frame from the camera
        _, frame = cap.read()

        # Get faces and landmarks using the face detector
        bboxes, landmarks = detector.detect(image=frame)

        if frame is not None:
            h, w, c = frame.shape
            tl = 1 or round(0.002 * (h + w) / 2) + 1  # Line and font thickness
            clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

            # Draw bounding boxes and landmarks on the frame
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            for key_point in landmarks:
                for id, point in enumerate(key_point):
                    cv2.circle(frame, tuple(point), tl + 1, clors[id], -1)


        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(
                frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        # Save the frame to the video
        video.write(frame)

        # Show the result in a window
        cv2.imshow("Face Detection", frame)

        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release video and camera, and close all OpenCV windows
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
