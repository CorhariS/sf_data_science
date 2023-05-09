
import pandas as pd
import pickle
from flask import Flask, request, jsonify

import sys, os
sys.path.append(os.path.join(os.path.abspath(''), '..', 'libs'))
import preparation

# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Тестовое сообщение. Сервер запущен!"
    return msg

@app.route('/predict', methods=['POST'])
def predict_func():    
    
    # загружаем модель из файла
    with open('../libs/data/model_abr.pkl', 'rb') as pkl_file:
        model = pickle.load(pkl_file)

    features = request.json
    df = pd.read_json(request.json, dtype={str})    
    df = preparation.preparate_file(df, method_clean=False)
    y_pred = model.predict(df)   
    
    return jsonify({'prediction': y_pred.tolist()})

if __name__ == '__main__':
    app.run('localhost', 5000)