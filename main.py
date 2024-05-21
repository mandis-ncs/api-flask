from flask import Flask, make_response, jsonify, request
import os
import pandas as pd
from utils import create_upload_folder, save_csv, delete_csv
from model import predictions

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

#add codigo rede neural
#pegar o valor da probabilidade
#devolver para kotlin no metodo GET
#apagar teste.csv (juntar a base original antes de apagar?)
#gerar requirements.txt

@app.route('/resultado', methods=['GET'])
def send_result():
    if not predictions or len(predictions) == 0:
        return jsonify({'error': 'Invalid data'}), 400
    # TypeError: Object of type ndarray is not JSON serializable
    # Erro ao retornar predictions direto no endpoint
    return jsonify(predictions)

app.run()