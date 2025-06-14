import cv2

# Load Haar cascade for face detection
face_cap = cv2.CascadeClassifier(
    "C:/Users/lenovo/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Start video capture
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break

    # Convert to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display video
    cv2.imshow("video_live", video_data)

    # Press 'a' to break the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
