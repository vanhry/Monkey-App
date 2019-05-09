from __future__ import division, print_function
import os
import numpy as np

from keras.models import model_from_json
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

monkey_labels = {
    0: "Mantled Howler",
    1: "Patas Monkey",
    2: "Bald Uakari",
    3: "Japanese Macaque",
    4: "Pygmy Marmoset",
    5: "White-headed Capuchin",
    6: "Silvery Marmoset",
    7: "Common Squirrel Monkey",
    8: "Black-headed Night Monkey",
    9: "Nilgiri Langur",
}

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

def load_model():
    json_file = open(os.path.join(os.getcwd(), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(os.getcwd(), "model.h5"))
    print("Loaded model from disk\n Check http://localhost:5000/")
    return loaded_model

model = load_model()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(160,160))

    # preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        pred_class = monkey_labels[np.argmax(preds)]
        return str(pred_class)

    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()