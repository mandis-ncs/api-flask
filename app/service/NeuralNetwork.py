from service.DataPreprocessor import DataPreprocessor
from models.config import UPLOADS_FOLDER, PKL_FOLDER, DB_FOLDER, CATEGORICAL_COLS
from models.ResponseEntity import ResponseEntity
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Dropout, Input
from sklearn.model_selection import KFold
from sqlalchemy.orm import Session
from typing import List
import joblib as jb
import pandas as pd
import numpy as np
import os

DATA_PATH = f'{DB_FOLDER}/dataset.csv'
MODEL_PATH = f'{DB_FOLDER}/autism_prediction_model.keras'

class NeuralNetworkService:
    def __init__(self, dense_units: int = 64, dropout_rate: float = 0.5, epochs: int = 30, batch_size: int = 32, verbose: int = 0, n_splits: int = 10):
        """
        Serviço de Rede Neural para classificação de dados de TEA. 

        :param dense_units: Número de unidades nas camadas densas (padrão: 64).
        :param dropout_rate: Taxa de dropout (padrão: 0.5).
        :param epochs: Número de epocas para o modelo (padrão: 30).
        :param batch_size: Tamanho do lote para o treinamento (padrão: 32).
        :param verbose: Nível de verbosidade durante o treinamento (padrão: 0).
        """

        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_splits = n_splits
        self.categorical_cols = CATEGORICAL_COLS

    
    def _create_model(self, input_shape: int, metrics: List[str] = ['accuracy']) -> Sequential:
        """
        Cria e compila um modelo de rede neural com camadas densas e dropout configuráveis.

        :param input_shape: Número de características da entrada.
        :param metrics: Lista de métricas para monitorar durante o treinamento e avaliação (padrão: ['accuracy']).
        :return: O modelo compilado.
        """

        try:
            model = Sequential([
                Input(shape=(input_shape,)),
                Dense(self.dense_units, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(self.dense_units, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(1, activation='sigmoid')
            ])

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

            return model
        
        except Exception as e:
            raise RuntimeError(f'Erro na criação do modelo da rede neural: {e}')
        
    
    def _cross_validate_model(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """
        Realiza a validação cruzada no modelo.

        :param X: DataFrame com as caracteristicas.
        :param y: Série com os rótulos.
        :return: Lista de acurácias para cada divisão.
        """
        try:
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            accuracies = []

            for train_index, test_index in kfold.split(X):
                model = self._create_model(X.shape[1])
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
                _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                accuracies.append(accuracy)

            return accuracies
        
        except Exception as e:
            raise RuntimeError(f'Erro na validação cruzada: {e}')


    def _preprocess_new_data(self, new_data: pd.DataFrame, label_encoders, onehot_encoder, scaler) -> pd.DataFrame:
        """
        Função para pré-processar a nova entrada de dados.

        :param new_data: DataFrame com os novos dados.
        :param label_encoders: Dicionário contendo os LabelEncoders para as colunas categóricas.
        :param onehot_encoder: O OneHotEncoder para transformar variáveis categóricas.
        :param scaler: O StandardScaler para normalizar os dados numéricos.
        :return: DataFrame pré-processado.
        """

        try:
            new_data_df = pd.DataFrame(new_data, columns=self.categorical_cols + ['Age_Mons', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])

            # Codificação de rótulos para variáveis categóricas
            for col, encoder in label_encoders.items():
                new_data_df[col] = encoder.transform(new_data_df[col])
            
            # Aplicar OneHotEncoder para variáveis categóricas
            categorical_data = onehot_encoder.transform(new_data_df[self.categorical_cols])
            categorical_df = pd.DataFrame(categorical_data, columns=onehot_encoder.get_feature_names_out(self.categorical_cols))

            # Normalizar a coluna Age_Mons
            new_data_df['Age_Mons'] = scaler.transform(new_data_df[['Age_Mons']])

            # Concatenar com o DataFrame original
            new_data_preprocessed = pd.concat([new_data_df.drop(self.categorical_cols, axis=1), categorical_df], axis=1)

            return new_data_preprocessed
        
        except Exception as e:
            raise ValueError(f'Erro ao pré-processar novos dados: {e}')


    def predict_from_csv(self, csv_path: str, label_encoders, onehot_encoder, scaler) -> np.ndarray:
        """
        Faz previsões a partir do arquivo teste.csv.

        :param csv_path: Caminho para o arquivo CSV com novos dados (assets/uploads/teste.csv).
        :param label_encoders: Dicionário contendo os LabelEncoders para as colunas categóricas.
        :param onehot_encoder: O OneHotEncoder para transformar variáveis categóricas.
        :param scaler: O StandardScaler para normalizar os dados numéricos.
        :return: Array com as previsões.
        """

        try:
            new_data = pd.read_csv(csv_path)
            new_data_preprocessed = self._preprocess_new_data(new_data, label_encoders, onehot_encoder, scaler)
            model = load_model(MODEL_PATH)
            pred_probabilities = model.predict(new_data_preprocessed)
            predictions = (pred_probabilities > 0.5).astype(int)
            
            return predictions
        
        except FileNotFoundError:
            raise FileNotFoundError(f'Arquivo {csv_path} não encontrado.')
        except Exception as e:
            raise RuntimeError(f'Erro ao fazer as previsões: {e}')
        
    
    def predict_from_db(self, record_id: int, db: Session) -> np.ndarray:
        """
        Faz previsões a partir de um registro no banco de dados.

        :param record_id: ID do registro no banco de dados.
        :param db: Sessão do banco de dados.
        :return: Array com as previsões.
        """
        try:
            # Buscar o registro no banco de dados
            record = db.query(ResponseEntity).filter(ResponseEntity.id == record_id).first()
            
            if record is None:
                raise ValueError(f'Registro com ID {record_id} não encontrado.')
            
            # Preparar os dados para o modelo
            record_data = {
                'A1': record.A1,
                'A2': record.A2,
                'A3': record.A3,
                'A4': record.A4,
                'A5': record.A5,
                'A6': record.A6,
                'A7': record.A7,
                'A8': record.A8,
                'A9': record.A9,
                'A10': record.A10,
                'Age_Mons': record.Age_Mons,
                'Sex': record.Sex,
                'Ethnicity': record.Ethnicity,
                'Jaundice': record.Jaundice,
                'Family_mem_with_ASD': record.Family_mem_with_ASD,
                'Class_ASD_Traits': record.Class_ASD_Traits
            }

            # Obter encoders e scaler
            label_encoders, onehot_encoder, scaler = self.get_encoders()

            # Pré-processar os dados
            data_df = pd.DataFrame([record_data])
            data_preprocessed = self._preprocess_new_data(
                data_df,
                label_encoders,
                onehot_encoder,
                scaler
            )
            
            # Fazer a previsão
            model = load_model(MODEL_PATH)
            pred_probabilities = model.predict(data_preprocessed)
            predictions = (pred_probabilities > 0.5).astype(int)

            return predictions
        
        except Exception as e:
            raise RuntimeError(f'Erro ao fazer a previsão a partir do banco de dados: {e}')
        
    
    def train_and_save_model(self):
        """
        Treina o modelo final com todos os dados e salva o modelo treinado.
        """

        try:
            preprocessor = DataPreprocessor(DATA_PATH)

            # Carregar e pré-processar os dados
            X, y = preprocessor.initialize()

            # Aplicar a validação cruzada
            accuracies = self._cross_validate_model(X, y)
            print(f'\nAcurácias da Validação Cruzada: \n{accuracies}\n')
            print(f'\nMédia da Acurácia: {np.mean(accuracies)}\n')

            # Treinar o modelo final em todos os dados e salvar o modelo treinado
            final_model = self._create_model(X.shape[1])
            final_model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            final_model.save(MODEL_PATH)

            print(f'\nModelo salvo em {MODEL_PATH}\n')

        except Exception as e:
            raise RuntimeError(f'Erro ao treinar e salvar o modelo: {e}')


    def get_encoders(self):
        """
        Carregar os encoders salvos.
        """

        try:
            label_encoders = {
                col: jb.load(f'{PKL_FOLDER}/label_encoder_{col}.pkl') for col in self.categorical_cols
            }
            onehot_encoder = jb.load(f'{PKL_FOLDER}/onehot_encoder.pkl')
            scaler = jb.load(f'{PKL_FOLDER}/scaler_Age_Mons.pkl')

            return label_encoders, onehot_encoder, scaler
    
        except Exception as e:
            raise RuntimeError(f'Erro durante a inicialização da rede neural: {e}')
