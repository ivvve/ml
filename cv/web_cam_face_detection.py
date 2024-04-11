import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(1)

try:
    while True:
        # capture frame-by-frame
        _, frame = video_capture.read()
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = face_detector(image_gray, 1)
        for face in detections:
            cv2.rectangle(frame, 
                          (face.left(), face.top()), 
                          (face.right(), face.bottom()), 
                          (0, 255, 0), 2)
        cv2.imshow("Video", frame)

        # quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # release resources
    video_capture.release()
    cv2.destroyAllWindows()
