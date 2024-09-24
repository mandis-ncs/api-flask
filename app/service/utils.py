from imblearn.over_sampling import SMOTE
from models.config import DB_FOLDER
import pandas as pd
import base64
import os


def create_upload_folder(folder_name):
    # Se pasta não existir, cria uma
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_csv(data, path):
    # Deletar arquivo existente se existir
    delete_csv(path)

    # Criar uma lista de dicionários com apenas os valores
    dict_data = [
        {
            'A1': item.A1,
            'A2': item.A2,
            'A3': item.A3,
            'A4': item.A4,
            'A5': item.A5,
            'A6': item.A6,
            'A7': item.A7,
            'A8': item.A8,
            'A9': item.A9,
            'A10': item.A10,
            'Age_Mons': item.Age_Mons,
            'Sex': item.Sex,
            'Ethnicity': item.Ethnicity,
            'Jaundice': item.Jaundice,
            'Family_mem_with_ASD': item.Family_mem_with_ASD,
            'Class_ASD_Traits': item.Class_ASD_Traits
        } for item in data
    ]
    
    # Criar DataFrame e salvar em CSV
    df = pd.DataFrame(dict_data)
    df.to_csv(path, index=False)


def delete_csv(path):
    # Deletar o arquivo CSV se existir
    if os.path.exists(path):
        os.remove(path)


def balance_data(X, y):
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled
    
    except Exception as e:
        raise RuntimeError(f'Erro no balanceamento dos dados: {e}')
    

def encode_img_base64(img_path: str) -> str:
    """Função para converter uma imagem para base64"""
    try:
        with open(img_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
        return base64_str
    except Exception as e:
        raise RuntimeError(f'Erro ao converter imagem para base64: {e}')


def filter_instances(qtd: int = 27):
    df = pd.read_csv(f'{DB_FOLDER}/dataset.csv')

    print(df.info())
    print()
    yes_instances = df[df['Class/ASD Traits'] == 'Yes'].sample(qtd)
    no_instances = df[df['Class/ASD Traits'] == 'No'].sample(qtd)

    print(yes_instances.head())
    print()
    print(no_instances.head())
    print()
    filtered_instances = pd.concat([yes_instances, no_instances])

    df_filtered = df.drop(filtered_instances.index)

    print(filtered_instances.info())
    print()
    print(df.info())

    df_filtered.to_csv(f'{DB_FOLDER}/dataset.csv', index=False)
    filtered_instances.to_csv(f'{DB_FOLDER}/filtered_dataset.csv', index=False)


                      
            