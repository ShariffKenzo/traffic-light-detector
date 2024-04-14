from flask import Flask, render_template, request
import base64
from subprocess import run

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import pygame

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['imageData']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    with open('captured_image.jpg', 'wb') as f:
        f.write(image_bytes)

    np.set_printoptions(suppress=True)


    model = load_model("keras_Model.h5", compile=False)

    class_names = open("labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open("captured_image.jpg").convert("RGB")

    # scale
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

 
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    print(class_name[2:])
    pygame.init()

    if ("Green" == class_name[2:].strip()):
        print(f"image is Green man")
        pygame.mixer.music.load('green_man_audio.mp3')  
        pygame.mixer.music.play()
    else:
        print(f"image is red man")
        pygame.mixer.music.load('red_man_audio.mp3')  
        pygame.mixer.music.play()
        
        return "success"

if __name__ == '__main__':
    app.run(debug=True)
