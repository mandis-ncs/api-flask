#REDE NEURAL MLP#

import numpy as np
from keras.api.layers import Dense, Dropout, Input # type: ignore
from keras.api.models import Sequential, load_model # type: ignore
from sklearn.model_selection import KFold
from data_preparation import load_and_preprocess_data
import joblib
import pandas as pd

# Função para criar o modelo
def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Função para realizar a validação cruzada
def cross_validate_model(X, y, n_splits=10):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kfold.split(X):
        model = create_model(X.shape[1])
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)
    return accuracies

# Função para pré-processar a nova entrada de dados
def preprocess_new_data(new_data, categorical_cols, label_encoders, onehot_encoder):
    new_data_df = pd.DataFrame([new_data], columns=categorical_cols + ['Age_Mons', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])

    # Codificação de rótulos para variáveis categóricas
    for col, encoder in label_encoders.items():
        new_data_df[col] = encoder.transform(new_data_df[col])

    # Aplicar OneHotEncoder para variáveis categóricas
    categorical_data = onehot_encoder.transform(new_data_df[categorical_cols]).toarray()
    categorical_df = pd.DataFrame(categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_cols))
    
    # Concatenar com o DataFrame original
    new_data_preprocessed = pd.concat([new_data_df.drop(categorical_cols, axis=1), categorical_df], axis=1)

    return new_data_preprocessed

# Carregar e pré-processar os dados
X, y = load_and_preprocess_data('dataset.csv')

# Aplicar a validação cruzada
accuracies = cross_validate_model(X, y, n_splits=10)
print("Acurácias da Validação Cruzada:", accuracies)
print("Média da Acurácia:", np.mean(accuracies))

# Treinar o modelo final em todos os dados e salvar o modelo treinado
final_model = create_model(X.shape[1])
final_model.fit(X, y, epochs=30, batch_size=32, verbose=0)
final_model.save('autism_prediction_model.keras')

# Carregar os encoders salvos
categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
label_encoders = {
    col: joblib.load(f'label_encoder_{col}.pkl') for col in categorical_cols
}
onehot_encoder = joblib.load('onehot_encoder.pkl')

# Nova entrada de dados (exemplo de uma criança que não possui autismo)
new_data = {
    'Sex': 'f',
    'Ethnicity': 'White European',
    'Jaundice': 'no',
    'Family_mem_with_ASD': 'no',
    'Age_Mons': 36,
    'A1': 0,
    'A2': 0,
    'A3': 0,
    'A4': 0,
    'A5': 0,
    'A6': 0,
    'A7': 0,
    'A8': 0,
    'A9': 0,
    'A10': 0
}

# Pré-processar a nova entrada de dados
new_data_preprocessed = preprocess_new_data(new_data, categorical_cols, label_encoders, onehot_encoder)

# Carregar o modelo salvo
model = load_model('autism_prediction_model.keras')

pred_probabilities = model.predict(new_data_preprocessed)
predictions = (pred_probabilities > 0.5).astype(int)

# Interpretação da previsão
print(predictions)