from service.DataPreprocessor import DataPreprocessor
from service.NeuralNetwork import NeuralNetworkService
from models.config import PLOT_FOLDER
import matplotlib.pyplot as plt
import numpy as np

def evaluate_accuracy_by_sample_size(data_path: str, start: int, end: int, step: int):
    try:
        preprocessor = DataPreprocessor(data_path)
        X_full, y_full = preprocessor.initialize()

        sample_sizes = range(start, end + 1, step)
        accuracies_list = []

        for size in sample_sizes:
            X_sample = X_full.sample(n=size, random_state=42)
            y_sample = y_full.loc[X_sample.index]

            nn_service = NeuralNetworkService()
            accuracies = nn_service._cross_validate_model(X_sample, y_sample)

            mean_accuracy = np.mean(accuracies)
            accuracies_list.append(mean_accuracy)

            print(f'Instâncias: {size}, Acurácia média: {mean_accuracy}')

        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, accuracies_list, marker='o', linestyle='-', color='b')
        plt.title('Acurácia em Função da Quantidade de Instâncias')
        plt.xlabel('Número de Instâncias')
        plt.ylabel('Acurácia Média')
        plt.xticks(sample_sizes)
        plt.grid(True)
        plt.savefig(f'{PLOT_FOLDER}/accuracy_by_sample_size.png', format='png', dpi=300)
        plt.show()

    except Exception as e:
        raise RuntimeError(f'Erro: \n{e}\n')