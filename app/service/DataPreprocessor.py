from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
from models.config import PKL_FOLDER, CATEGORICAL_COLS, LABEL_COL, DB_FOLDER
import pandas as pd
import numpy as np
import joblib as jb
import os

class DataPreprocessor:
    def __init__(self, filepath: str):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.scaler = StandardScaler()
        self.categorical_cols = CATEGORICAL_COLS
        self.label_col = LABEL_COL
        self.label_encoders = {}
        self.categorical_data = None
        self.categorical_df = None
        self.data_path = filepath
        self.data_model = None
        self.data_preprocessed = None

    def _validate_columns(self):
        """
        Valida se todas as colunas necessárias estão presentes no DataFrame.
        """

        missing_cols = set(self.categorical_cols + ['Age_Mons']).difference(self.data_model.columns)
        
        if missing_cols:
            raise ValueError(f'\nColunas faltantes no DataFrame: {missing_cols}')

    def _preprocess_labels(self):
        try:
            self.data_model = pd.read_csv(self.data_path)
            self._validate_columns()

            # Codificação dos rótulos da classe alvo
            self.data_model[self.label_col] = self.label_encoder.fit_transform(self.data_model[self.label_col])

            # Selecionar colunas categóricas para codificação one-hot
            self.label_encoders = {col: LabelEncoder() for col in self.categorical_cols}

            # Aplicar LabelEncoder para variáveis categóricas
            for col in self.categorical_cols:
                self.data_model[col] = self.label_encoders[col].fit_transform(self.data_model[col])

            # Salvar os label encoders
            for col, encoder in self.label_encoders.items():
                jb.dump(encoder, f'{PKL_FOLDER}/label_encoder_{col}.pkl')

        except Exception as e:
            raise RuntimeError(f'\nErro na codificação das labels da classe alvo: \n{e}\n')


    def _preprocess_categorical(self):
        try:
            # Aplicar OneHotEncoder para variáveis categóricas
            self.categorical_data = self.onehot_encoder.fit_transform(self.data_model[self.categorical_cols])

            # Salvar o OneHotEncoder
            jb.dump(self.onehot_encoder, f'{PKL_FOLDER}/onehot_encoder.pkl')

            # Criar um DataFrame com os dados categóricos codificados
            self.categorical_df = pd.DataFrame(self.categorical_data, columns=self.onehot_encoder.get_feature_names_out(self.categorical_cols))

            # Concatenar com o DataFrame original (sem as colunas categóricas originais)
            self.data_preprocessed = pd.concat([self.data_model.drop(self.categorical_cols, axis=1), self.categorical_df], axis=1)

        except Exception as e:
            raise RuntimeError(f'\nErro no processamento das variáveis categóricas: \n{e}\n')


    def _normalize_age(self):
        try:
            # Normalizar a coluna Age_Mons
            self.data_preprocessed['Age_Mons'] = self.scaler.fit_transform(self.data_preprocessed[['Age_Mons']])

            # Salvar o scaler
            jb.dump(self.scaler, f'{PKL_FOLDER}/scaler_Age_Mons.pkl')

        except Exception as e:
            raise RuntimeError(f'\nErro ao tentar normalizar a coluna Age_Mons: \n{e}\n')


    def initialize(self):
        try:
            self._preprocess_labels()
            self._preprocess_categorical()
            self._normalize_age()

            # Separar os recursos (features) e os rótulos (labels)
            X = self.data_preprocessed.drop('Class/ASD Traits', axis=1)
            y = self.data_preprocessed['Class/ASD Traits']

            return X, y
        
        except Exception as e:
            raise RuntimeError(f'\nErro no pré-processamento dos dados: \n{e}\n')
