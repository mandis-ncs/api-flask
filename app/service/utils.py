from imblearn.over_sampling import SMOTE
import pandas as pd
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