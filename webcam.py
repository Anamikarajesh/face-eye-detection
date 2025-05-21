from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(BASE_DIR, 'haarcascade_eye.xml')

face_detector = cv2.CascadeClassifier(face_cascade_path)
eye_detector = cv2.CascadeClassifier(eye_cascade_path)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
