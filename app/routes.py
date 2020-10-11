#-*- coding: UTF-8 -*-
#!/usr/bin/env python
#BackEnd
from flask import jsonify
import os, sys, json
from PIL import Image as img
from flask import request
from numpy import asarray
#ML
import pandas as pd
import skimage.io as io
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

def get_class_string_from_index(index):
    for name, number in dict_name.items():
        if number == index:
            return name

def runNeoro(data):
    new_model = tf.keras.models.load_model(filepath='/Users/nikita/Desktop/Папки & документы/Python/', compile=True)
    new_model.summary()
    dict_name = {'Hyundai Solaris sedan': 0,
                 'KIA Rio sedan': 1,
                 'SKODA OCTAVIA sedan': 2,
                 'Volkswagen Polo sedan': 3,
                 'Volkswagen_Tiguan': 4}

    prediction_scores = new_model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("Predicted label: " + str(get_class_string_from_index(predicted_index)))
    print('Probability: ' + str(np.max(np.round(prediction_scores * 100, 1)[0])))

def img2data(img):
    image = Image.open(img)
    image.resize((299, 299))
    return asarray(image.load())

from app import app
@app.route('/')
@app.route('/api')
def index():
    return "hello world"
# api получаем картинку оправляем data
#дкоратор  описывает  маршруты, далее фунция обработка
@app.route("/api/cars", method=['GET'])
def get_car():
    #получаем данные из запроса
    im = request.data()
    #обрабатываем картинк
    data = img2data(im)
    runNeoro(im)
    # Возращаем результат ввиде json
    return json.dump(data)
    # Весь API

@app.route('/api/marketplace')
def get_marketpalce():
    pass

