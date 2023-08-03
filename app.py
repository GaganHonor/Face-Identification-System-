from flask import Flask, render_template, Response, jsonify
import cv2
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

recognized_faces = []  # Global list to store recognized faces

@app.route('/')
def index():
    return render_template('index.html')

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model():
    faces = []
    ids = []

    id = 1
    names = {0: "None"}

    training_dirs = [d for d in os.listdir('training_images') if os.path.isdir(os.path.join('training_images', d))]

    for dir in training_dirs:
        training_image_path = [os.path.join(f'training_images/{dir}', f) for f in os.listdir(f'training_images/{dir}')]

        for image_path in training_image_path:
            img = Image.open(image_path).convert('L')
            img_numpy = np.array(img, 'uint8')
            faces.append(img_numpy)
            ids.append(id)

        names[id] = dir
        id += 1

    face_recognizer.train(faces, np.array(ids))
    return names

names = train_model()

def gen_frames():  
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])

                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(
                           frame, 
                           str(id), 
                           (x+5,y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (255,255,255), 
                           2
                          )
                cv2.putText(
                           frame, 
                           str(confidence), 
                           (x+5,y+h-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (255,255,0), 
                           1
                          ) 
                
                global recognized_faces
                recognized_faces.append(id)  # Add recognized face to global list

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faces')
def faces():
    global recognized_faces
    faces_to_return = list(recognized_faces)  # Copy recognized faces to local list
    recognized_faces = []  # Clear the global list
    return jsonify(faces=faces_to_return)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
