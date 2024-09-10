from service.NeuralNetwork import NeuralNetworkService, DATA_PATH
from service.DataPreprocessor import DataPreprocessor
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_by_epochs():
    try:
        # Armazena as épocas e as acurácias médias correspondentes
        epochs_list = list(range(10, 201, 10))
        accuracies_list = []

        # Inicializa o DataPreprocessor para carregar os dados
        preprocessor = DataPreprocessor(DATA_PATH)
        X, y = preprocessor.initialize()

        for epochs in epochs_list:
            nn_service = NeuralNetworkService(epochs=epochs)

            # Realiza a validação cruzada e obtém as acurácias
            accuracies = nn_service._cross_validate_model(X, y)

            # Calcula a acurácia média e adiciona à lista
            mean_accuracy = np.mean(accuracies)
            accuracies_list.append(mean_accuracy)
            
            print(f'Épocas: {epochs}, Acurácia média: {mean_accuracy}')

        # Gera o gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, accuracies_list, marker='o', linestyle='-', color='b')
        plt.title('Acurácia Média vs. Número de Épocas')
        plt.xlabel('Número de Épocas')
        plt.ylabel('Acurácia Média')
        plt.xticks(epochs_list)
        plt.grid(True)
        plt.savefig('./grafico_acuracia_epocas.png', format='png', dpi=300)
        plt.show()

    except Exception as e:
        raise RuntimeError(f'Erro: \n{e}\n')