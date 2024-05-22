from flask import Flask, make_response, jsonify, request
import os
import pandas as pd
from utils import create_upload_folder, save_csv, delete_csv
from model import predictions
import numpy as np

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

#add codigo rede neural, no lugar de 'new data' vai o teste.csv
#apagar teste.csv
#gerar requirements.txt
#git ignore

@app.route('/resultado', methods=['GET'])
def send_result():
    if not predictions or len(predictions) == 0:
        return jsonify({'error': 'Invalid data'}), 400
    
    # Extract the first value from predictions
    result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]

    # Return the result as a JSON response
    return jsonify({"resultado": str(result_value)})

app.run()