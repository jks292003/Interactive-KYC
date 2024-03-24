import cv2
import sqlite3
from io import BytesIO
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('faces.db')
c = conn.cursor()
#c.execute("drop table faces")

# Create table to store face images
c.execute('''CREATE TABLE IF NOT EXISTS faces
             (id INTEGER PRIMARY KEY, name TEXT, image BLOB)''')
def resize_image(img, height):
    """Resize image to specified height while preserving aspect ratio."""
    ratio = height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * ratio), height))
# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_face(name):
    # Read image from webcam
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 1:
        # Extract the face region
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y + w, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
        
        # Convert image to bytes
        image_bytes = cv2.imencode('.jpg', roi_color)[1].tobytes()

        # Save the face image to the database
        c.execute("INSERT INTO faces (name, image) VALUES (?, ?)", (name, image_bytes))
        conn.commit()
        print("Face captured and stored successfully!")
    else:
        print("No face detected or multiple faces detected.")
    cv2.imshow("test",frame)
    
def show_all_faces():
    # Select all rows from the faces table
    c.execute("SELECT * FROM faces")
    rows = c.fetchall()
    images = []
    max_height = 0
    # Print the rows
    for row in rows:
        print("ID:", row[0])
        print("Name:", row[1])
        image_data = np.frombuffer(row[2], dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        max_height = max(max_height, img.shape[0])
        images.append(img)
    resized_images = [resize_image(img, max_height) for img in images]
    combined_image = np.hstack(resized_images)
    cv2.imshow("All Faces", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    name = input("Enter the name of the person: ")
    
    capture_face(name)
    show_all_faces()
