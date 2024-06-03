from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
import numpy as np 
from model import predict_from_csv
from utils import create_upload_folder, save_csv, delete_csv

# Instanciar flask
app = Flask('__name__')

# Receber respostas
@app.route('/respostas', methods=['POST'])
def upload_data():
    # Cria pasta uploads se nao existir
    folder_name = 'uploads'
    create_upload_folder(folder_name)

    # Request do json, guarda no array
    # Se tiver vazio ou com outro nome que não 'responses' dá erro
    data = request.get_json()
    if not data or 'responses' not in data:
        return jsonify({'error': 'Invalid data'}), 400
        
    responses = data['responses']
    csv_path = os.path.join(folder_name, 'teste.csv')

    save_csv(responses, csv_path)
    return responses

@app.route('/resultado', methods=['GET'])
def send_result():
    folder_name = 'uploads'
    csv_path = os.path.join(folder_name, 'teste.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'CSV file not found'}), 400

    categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    label_encoders = {
        col: joblib.load(f'label_encoder_{col}.pkl') for col in categorical_cols
    }
    onehot_encoder = joblib.load('onehot_encoder.pkl')
    scaler = joblib.load('scaler_Age_Mons.pkl')
    model_path = 'autism_prediction_model.keras'

    predictions = predict_from_csv(csv_path, model_path, categorical_cols, label_encoders, onehot_encoder, scaler)

    result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]

    # Opcionalmente, excluir o arquivo CSV após a predição
    delete_csv(csv_path)

    return jsonify({"resultado": str(result_value)})

if __name__ == '__main__':
    app.run()
