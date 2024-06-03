import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Codificação dos rótulos da classe alvo
    label_encoder = LabelEncoder()
    data['Class/ASD Traits '] = label_encoder.fit_transform(data['Class/ASD Traits '])
    
    # Selecionar colunas categóricas para codificação one-hot
    categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    
    # Aplicar LabelEncoder para variáveis categóricas
    for col in categorical_cols:
        data[col] = label_encoders[col].fit_transform(data[col])

    # Salvar os label encoders
    for col, encoder in label_encoders.items():
        joblib.dump(encoder, f'label_encoder_{col}.pkl')

    # Aplicar OneHotEncoder para variáveis categóricas
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_data = onehot_encoder.fit_transform(data[categorical_cols])
    
    # Salvar o OneHotEncoder
    joblib.dump(onehot_encoder, 'onehot_encoder.pkl')

    # Criar um DataFrame com os dados categóricos codificados
    categorical_df = pd.DataFrame(categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_cols))

    # Concatenar com o DataFrame original (sem as colunas categóricas originais)
    data_preprocessed = pd.concat([data.drop(categorical_cols + ['Who completed the test', 'Case_No'], axis=1), categorical_df], axis=1)

    # Normalizar a coluna Age_Mons
    scaler = StandardScaler()
    data_preprocessed['Age_Mons'] = scaler.fit_transform(data_preprocessed[['Age_Mons']])

    # Salvar o scaler
    joblib.dump(scaler, 'scaler_Age_Mons.pkl')

    # Separar os recursos (features) e os rótulos (labels)
    X = data_preprocessed.drop('Class/ASD Traits ', axis=1)
    y = data_preprocessed['Class/ASD Traits ']

    return X, y  # Retorna os dados preparados