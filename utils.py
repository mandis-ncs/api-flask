import os
import pandas as pd

def create_upload_folder(folder_name):
    # Se pasta não existir, cria uma
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

