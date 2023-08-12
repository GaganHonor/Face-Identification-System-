# Face-Identification-System-
Face Recognition System by @GaganHonor 
🤖
// code 

<p align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" width ="250">
  </a>
</p>

<div>

 

 
</div>

## Face Recognition System is a web app that allows you to identify and recognize faces in images using OpenCV and Python. You can upload an image or use your webcam to capture a photo, and the app will detect and display the faces in the image along with their names if they are known. You can also train the app to recognize new faces by providing some sample images of the person you want to add.

## Table of Contents
Installation
Usage
Screenshots
Code
Contributing
License
Installation

## To install and run Face Recognition System, you need to have Python 3 and OpenCV installed on your computer. You also need to install the face_recognition library, which is a wrapper for the dlib library that provides face detection and recognition features. To install these dependencies, you can use the following commands:

## INSTALLATION

To install and run Face Recognition System, you need to have Python 3 and OpenCV installed on your computer. You also need to install the face_recognition library, which is a wrapper for the dlib library that provides face detection and recognition features. To install these dependencies, you can use the following commands:
````bash
pip install -r requirements.txt
pip install opencv-python flask numpy pillow
pip install flask opencv-python numpy
python app.py
````
You also need to clone this repository or download the source code as a ZIP file. To clone the repository, you can use the following command:
````bash
git clone https://github.com/GaganHonor/Face-Identification-System-
````

## Screenshots
Here are some screenshots of Face Recognition System in action:



<img src="readmeimg/Screenshot 2023-08-03 161338.jpg" alt="Screenshot 1" class="image">
<img src="readmeimg/Screenshot 2023-08-03 161407.jpg" alt="Screenshot 2" class="image">

CodeFace Recognition System uses OpenCV and face_recognition libraries to perform face detection and recognition tasks. Here is a brief explanation of how they work:
- OpenCV is an open source computer vision library that provides various functions and algorithms for image processing, video analysis, object detection, face recognition, etc.
- face_recognition is a Python library that wraps around dlib's state-of-the-art face recognition algorithms. It uses a deep learning model called ResNet-34 with 99.4% accuracy on Labeled Faces in The Wild (LFW) benchmark.
- To detect faces in an image, face_recognition uses a pre-trained histogram of oriented gradients (HOG) feature detector that scans the image at multiple scales and returns bounding boxes for each face.
- To recognize faces in an image, face_recognition uses a pre-trained convolutional neural network (CNN) that extracts 128 features from each face and compares them with the features of known faces using a distance metric. The face with the smallest distance is considered the most similar one.
Here is a snippet of code that shows how Face Recognition System uses these libraries:
```bash
# Import libraries
import cv2
import face_recognition

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load the face recognizer for face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the images of known faces and their names
known_faces = []
known_names = []
for filename in os.listdir('images'):
    # Extract the name from the filename
    name = filename.split('_')[0]
    # Load the image as a numpy array
    image = face_recognition.load_image_file('images/' + filename)
    # Encode the face into a 128-dimensional vector
    encoding = face_recognition.face_encodings(image)[0]
    # Append the encoding and the name to the lists
    known_faces.append(encoding)
    known_names.append(name)

# Train the face recognizer with the known faces and names
face_recognizer.train(known_faces, known_names)

# Load the image or capture the photo to be processed
image = ...

# Convert the image to grayscale for better performance
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image using the cascade classifier
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Loop over each detected face
for (x, y, w, h) in faces:
    # Crop the face from the image
    face = image[y:y+h, x:x+w]
    # Encode the face into a 128-dimensional vector
    encoding = face_recognition.face_encodings(face)[0]
    # Predict the name of the face using the face recognizer
    name = face_recognizer.predict(encoding)
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Put the name below the face
    cv2.putText(image, name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
```
**If you need any more modes in repo or If you find out any bugs, mention in [@gaganhonor ](https://www.telegram.dog/)**

## Contribution
Face Recognition System is an open source project and welcomes contributions from anyone who is interested in improving it. If you want to contribute to this project, you can follow these steps:

Fork this repository on GitHub
Clone your forked repository on your local machine
Create a new branch for your feature or bug fix
Make your changes and commit them with a descriptive message
Push your changes to your forked repository
Create a pull request from your forked repository to this repository
Wait for your pull request to be reviewed and merged
Please make sure that your code follows the PEP 8 style guide for Python code and that you document your code properly.

### Features
To run Face Recognition System, you need to launch the app.py file using Python. You can use the following command:
````bash
python app.py
````
Copy
This will start a local web server on port 5000. You can then open your web browser and go to http://localhost:5000 to access the app.

To use the app, you need to have some images of faces stored in the images folder. The app will use these images to train the face recognizer and assign names to the faces. The images should be named as {person_name}_{number}.jpg, where person_name is the name of the person in the image and number is a sequential number. For example, alice_1.jpg, alice_2.jpg, bob_1.jpg, bob_2.jpg, etc.

To upload an image or capture a photo using your webcam, you can use the buttons on the app’s homepage. The app will then process the image and display the results on a new page. You will see the original image with rectangles around the detected faces and their names below them. You can also download the processed image by clicking on the Download button.

To add a new face to the app’s database, you can use the Add Face button on the app’s homepage. This will take you to a new page where you can enter the name of the person you want to add and upload some sample images of their face. The app will then train the face recognizer with these images and add them to the images folder.




Report Bugs, Give Feature Requests There..   

### Credits


### Licence
[![GNU GPLv3 Image](https://www.gnu.org/graphics/gplv3-127x51.png)](http://www.gnu.org/licenses/gpl-3.0.en.html)  

[FACE-IDENTIFICATION-SYSTEM](https://github.com/GaganHonor/Face-Identification-System-/) is Free Software: You can use, study share and improve it at your
will. Specifically you can redistribute and/or modify it under the terms of the
[GNU General Public License](https://www.gnu.org/licenses/gpl.html) as
published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. 

##

   **Star this Repo if you Liked it ⭐⭐⭐**

