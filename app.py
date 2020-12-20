from flask import Flask, render_template, request, send_from_directory

from keras.models import load_model
from keras.preprocessing import image

import numpy as np

app = Flask(__name__)
COUNT = 1


def load_model_(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    saved_model = load_model('sav/vggmodel.h5')
    output = saved_model.predict(img)[0]
    if output[0] > output[1]:
        res = f'Benign with probability:\t{round(output[0], 2)}'
    else:
        res = f'Malignant with probability:\t{round(output[1], 2)}'
    return res


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global COUNT

    img = request.files['image']
    img.save('static/{}.jpg'.format(COUNT))
    res = load_model_('static/{}.jpg'.format(COUNT))

    return render_template('data.html', output=res)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT))


if __name__ == "__main__":
    app.run(debug=True, port = '5000')
