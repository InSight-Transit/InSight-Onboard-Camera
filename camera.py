import cv2
import requests
import tempfile

# Corrected filename for the Haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

API_URL = "http://127.0.0.1:8000/api/search"

while True:
    activate = False
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0 and activate is True:  # If at least one face is detected
        print(f"Detected {len(faces)} face(s). Taking screenshot and sending API request...")

        # Save the frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            cv2.imwrite(temp_file.name, frame)
            print(f"Temporary screenshot saved at {temp_file.name}")

            # Send API request with the temporary file
            try:
                with open(temp_file.name, 'rb') as image_file:
                    files = {'file': image_file}
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        print("API request successful.")
                    else:
                        print(f"API request failed with status code {response.status_code}.")
            except Exception as e:
                print(f"Error sending API request: {e}")

    # Display the resulting frame
    feed = cv2.flip(gray, 1)
    cv2.imshow('frame', feed)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()