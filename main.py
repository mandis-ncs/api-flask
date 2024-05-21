from flask import Flask, make_response, jsonify, request
from bd import carros
import os
import pandas as pd

#instanciar flask
app = Flask('__name__')
app.json.sort_keys = False


#UTILS Functions
def create_upload_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_csv(data, path):
    # Deletar arquivo existente se existir
    delete_csv(path)

    # Criar um novo arquivo CSV com os dados fornecidos
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def delete_csv(path):
    # Deletar o arquivo CSV se existir
    if os.path.exists(path):
        os.remove(path)


#MAIN  functions
folder_name = 'uploads'
create_upload_folder(folder_name)


@app.route('/respostas', methods=['POST'])
def upload_data():
    data = request.get_json()
    if not data or 'responses' not in data:
        return jsonify({'error': 'Invalid data'}), 400
        
    responses = data['responses']
    csv_path = os.path.join(folder_name, 'teste.csv')

    # Salvar o array de respostas em um arquivo CSV
    save_csv(responses, csv_path)
    return responses







app.run()