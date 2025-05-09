import cv2
import requests
import tempfile
from playsound import playsound
import time

last_request_time = 0
request_interval = 3

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

API_URL = "http://localhost:8000/api/search"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    current_time = time.time()
    if len(faces) > 0 and (current_time - last_request_time >= request_interval):  # If at least one face is detected
        last_request_time = current_time
        print(f"Detected {len(faces)} face(s). Sending API request...")

        for i, (x, y, w, h) in enumerate(faces):
            # Crop the face region from the frame
            face_region = frame[y:y+h, x:x+w]

            # Save the cropped face to a temporary file
            with tempfile.NamedTemporaryFile(suffix=f"_face_{i}.png", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, face_region)
                print(f"Face saved at {temp_file.name}")

                # Send API request with the cropped face
                try:
                    with open(temp_file.name, 'rb') as image_file:
                        files = {'file': (f"face_{i}.png", image_file, 'image/png')}
                        response = requests.post(API_URL, files=files, timeout=10)
                        if response.status_code == 200:
                            print(f"API request for face {i} successful.")
                            response_data = response.json()
                            account_id = response_data.get('Account ID')
                            print(response_data)
                            if account_id:
                                print(f"User {account_id} successful.")
                                playsound("./sounds/accept.mp3")
                            else:
                                print("Account ID not found in the response.")
                                playsound("./sounds/error.mp3")
                        else:
                            print(f"API request for face {i} failed with status code {response.status_code}.")
                            playsound("./sounds/error.mp3")
                except requests.exceptions.Timeout:
                    print(f"API request for face {i} timed out.")
                    playsound("./sounds/error.mp3")
                except Exception as e:
                    print(f"Error sending API request for face {i}: {e}")
                    playsound("./sounds/error.mp3")


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    feed = cv2.flip(frame, 1)
    cv2.imshow('frame', feed)

    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()