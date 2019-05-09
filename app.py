from __future__ import division, print_function
import os
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import model_from_json
from keras.preprocessing import image

# Flask utils
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

# Model saved with Keras model.save()
# MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

# print('Model loaded. Check http://127.0.0.1:5000/')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

def build_model():
    json_file = open(os.path.join(os.getcwd(), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(os.getcwd(), "model.h5"))
    print("Loaded model from disk\nCheck http://localhost:5000/")
    return loaded_model

model = build_model()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(160,160))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
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

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = monkey_labels[np.argmax(preds)]  # ImageNet Decode
                      # Convert to string
        return str(pred_class)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent

    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()