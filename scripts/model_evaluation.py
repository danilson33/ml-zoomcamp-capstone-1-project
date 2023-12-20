from PIL import Image
import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten
from flask import Flask
from flask import request
from flask import jsonify

import warnings

warnings.filterwarnings('ignore')

IMG_SAVE_PATH = '/tmp'
MODEL_FILE = "/app/artifacts/vgg16_model.h5"
IMG_SIZE = 224
NUM_COLOR_BITS = 3
NUM_CLASSES = 48
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)

app = Flask('font-type-prediction')

def get_model(model_input_shape, num_classes):
    base_model = VGG16
    base_model = base_model(include_top=False, weights="imagenet", input_shape = model_input_shape)
    base_model.trainable = False

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))

    return model

def get_label_by_class(id):
    mapping = {
        0: 'Agency',
        1: 'Akzidenz Grotesk',
        2: 'Algerian',
        3: 'Arial',
        4: 'Baskerville',
        5: 'Bell MT',
        6: 'Bembo',
        7: 'Bodoni',
        8: 'Book Antiqua',
        9: 'Brandish',
        10: 'Calibry',
        11: 'Californian FB',
        12: 'Calligraphy',
        13: 'Calvin',
        14: 'Cambria',
        15: 'Candara',
        16: 'Century',
        17: 'Comic Sans MS',
        18: 'Consolas',
        19: 'Corbel',
        20: 'Courier',
        21: 'Didot',
        22: 'Elephant',
        23: 'Fascinate',
        24: 'Franklin Gothic',
        25: 'Futigre',
        26: 'Futura',
        27: 'Garamond',
        28: 'Georgia',
        29: 'Gill Sans',
        30: 'Helvetica',
        31: 'Hombre',
        32: 'Lato',
        33: 'LCD Mono',
        34: 'Lucida Bright', 
        35: 'Monotype Corsiva',
        36: 'Mrs Eaves',
        37: 'Myriad',
        38: 'Nasalization',
        39: 'News Gothic',
        40: 'Palatino linotype',
        41: 'Perpetua',
        42: 'Rockwell',
        43: 'Sabon',
        44: 'Snowdrift Regular',
        45: 'Steppes',
        46: 'Times New Roman',
        47: 'Verdana'
    }
    
    return mapping.get(id)

def predict_class(path, target_size, model):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    predicts = model.predict(x)

    return np.argmax(predicts[0])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(request.files['file'])
        path = os.path.join(IMG_SAVE_PATH, 'test.jpg')
        img.save(path)
        logger = logging.getLogger("Font type model evaluation")
        logging.basicConfig(level=logging.INFO)

        logger.info("Loading model")
        model = get_model((IMG_SIZE, IMG_SIZE, NUM_COLOR_BITS), NUM_CLASSES)
        model.load_weights(MODEL_FILE)

        logger.info("Starting prediction")
        predicted_class = predict_class(path, TARGET_SIZE, model)
        predicted_label = get_label_by_class(predicted_class)
        logger.info(f"Result class: {predicted_class}, label: {predicted_label}")

        result = {
            "status": 'success',
            "font_type": predicted_label
        }

    except Exception as e:
        result = {
            "status": 'error',
            "message": str(e)
        }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
