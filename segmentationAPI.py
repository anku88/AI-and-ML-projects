
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import PIL
from PIL import Image

app = Flask(__name__)

# model path
path = r"C:\Users\Anku\Downloads\UNET_TASK\flask_deployment\artifacts\H&E.h5"

def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return K.mean(numerator / (denominator + tf.keras.backend.epsilon()))

def loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    eps = tf.keras.backend.epsilon()
    dice_log = tf.math.log(dice + eps)
    return bce - dice_log

model = tf.keras.models.load_model(path, custom_objects={'loss': loss, 'dice_coefficient': dice_coefficient})



img_size = 1000
num_classes = 2


def preprocess_image(img):
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    img = tf.image.resize(img, (img_size, img_size))
    return img

def segment_image(img):
    global predicted_classes  # Access the global variable
    # Convert RGB image to grayscale
    img = tf.image.rgb_to_grayscale(img)
    # Preprocess the image

    test_img = preprocess_image(img)
    test_img = tf.expand_dims(test_img, axis=0)

    pred = model.predict(test_img)
    predicted_classes = np.argmax(pred[0], axis=-1)



# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')


# Define the route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global predicted_classes  # Access the global variable
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Open the TIFF image using PIL and convert it to RGB format
        img = PIL.Image.open(file)
        img = img.convert('RGB')
        # Convert the image to a NumPy array and resize it
        img = np.array(img)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.uint8)
        # Perform segmentation on the input image
        segment_image(img)
        # Convert the output image to a PNG file for display in the HTML page
        output = io.BytesIO()
        plt.imsave(output, np.reshape(predicted_classes, (img_size, img_size)), cmap='gray', format='png')
        output.seek(0)
        output_str = base64.b64encode(output.getvalue()).decode('ascii')
        output.close()  # Close the output file
        
        # Return the HTML page with the segmented image
        return render_template('home.html', output=output_str)
    
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)