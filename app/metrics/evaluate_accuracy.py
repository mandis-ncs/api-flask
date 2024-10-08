from service.NeuralNetwork import NeuralNetworkService, DATA_PATH
from service.DataPreprocessor import DataPreprocessor
from service.utils import encode_img_base64
from models.config import PLOT_FOLDER
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_by_epochs(show_plot: bool = False):
    try:
        # Armazena as épocas e as acurácias médias correspondentes
        epochs_list = list(range(10, 101, 10))
        accuracies_list = []

        # Inicializa o DataPreprocessor para carregar os dados
        preprocessor = DataPreprocessor(DATA_PATH)
        X, y = preprocessor.initialize()

        results = []

        graph_path = f'{PLOT_FOLDER}/accuracy_by_epochs_by_step_10.png'


        for epochs in epochs_list:
            nn_service = NeuralNetworkService(epochs=epochs)

            # Realiza a validação cruzada e obtém as acurácias
            accuracies = nn_service._cross_validate_model(X, y)

            # Calcula a acurácia média e adiciona à lista
            mean_accuracy = round(np.mean(accuracies), 4)
            accuracies_list.append(mean_accuracy)
            
            print(f'Épocas: {epochs}, Acurácia média: {mean_accuracy}')
            results.append({'epocas': epochs, 'acuracia_media': mean_accuracy})

        # Gera o gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, accuracies_list, marker='o', linestyle='-', color='b')
        plt.title('Acurácia Média vs. Número de Épocas')
        plt.xlabel('Número de Épocas')
        plt.ylabel('Acurácia Média')
        plt.xticks(epochs_list)
        plt.grid(True)
        plt.savefig(graph_path, format='png', dpi=300)
        
        if show_plot:
            plt.show()

        return{
            'results': results,
            'graph_base64': encode_img_base64(graph_path)
        }
            
    except Exception as e:
        raise RuntimeError(f'Erro: \n{e}\n')


def evaluate_accuracy_by_sample_size(data_path: str, start: int, end: int, step: int, show_plot: bool = False):
    try:
        preprocessor = DataPreprocessor(data_path)
        X_full, y_full = preprocessor.initialize()

        sample_sizes = range(start, end + 1, step)
        accuracies_list = []

        results = []

        graph_path = f'{PLOT_FOLDER}/accuracy_by_sample_size.png'

        for size in sample_sizes:
            X_sample = X_full.sample(n=size, random_state=42)
            y_sample = y_full.loc[X_sample.index]

            nn_service = NeuralNetworkService()
            accuracies = nn_service._cross_validate_model(X_sample, y_sample)

            mean_accuracy = round(np.mean(accuracies), 4)
            accuracies_list.append(mean_accuracy)

            print(f'Instâncias: {size}, Acurácia média: {mean_accuracy}')
            results.append({'instancias': size, 'acuracia_media': mean_accuracy})

        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, accuracies_list, marker='o', linestyle='-', color='b')
        plt.title('Acurácia em Função da Quantidade de Instâncias')
        plt.xlabel('Número de Instâncias')
        plt.ylabel('Acurácia Média')
        plt.xticks(sample_sizes)
        plt.grid(True)
        plt.savefig(graph_path, format='png', dpi=300)
        
        if show_plot:
            plt.show()

        return{
            'results': results,
            'graph_base64': encode_img_base64(graph_path)
        }

    except Exception as e:
        raise RuntimeError(f'Erro: \n{e}\n')
    
def accuracy_x_epochs_x_samples(show_plot: bool = False):
    try:
        # Definir épocas
        epochs_list = [20, 40, 60, 80, 100]
        
        # Inicializar DataPreprocessor para carregar dados completos
        preprocessor = DataPreprocessor(DATA_PATH)
        X_full, y_full = preprocessor.initialize()
        
        # Listas para armazenar acurácia por frações (1/3, 2/3, 3/3)
        accuracies_1_3 = []
        accuracies_2_3 = []
        accuracies_3_3 = []
        
        # Treinamento para diferentes frações da base de dados
        for epochs in epochs_list:
            # 1/3 da base
            X_sample_1_3 = X_full.sample(frac=1/3, random_state=42)
            y_sample_1_3 = y_full.loc[X_sample_1_3.index]
            nn_service_1_3 = NeuralNetworkService(epochs=epochs)
            acc_1_3 = np.mean(nn_service_1_3._cross_validate_model(X_sample_1_3, y_sample_1_3))
            accuracies_1_3.append(acc_1_3)

            # 2/3 da base
            X_sample_2_3 = X_full.sample(frac=2/3, random_state=42)
            y_sample_2_3 = y_full.loc[X_sample_2_3.index]
            nn_service_2_3 = NeuralNetworkService(epochs=epochs)
            acc_2_3 = np.mean(nn_service_2_3._cross_validate_model(X_sample_2_3, y_sample_2_3))
            accuracies_2_3.append(acc_2_3)

            # 3/3 da base (todos os dados)
            nn_service_3_3 = NeuralNetworkService(epochs=epochs)
            acc_3_3 = np.mean(nn_service_3_3._cross_validate_model(X_full, y_full))
            accuracies_3_3.append(acc_3_3)
        
        # Plotar o gráfico com diferentes curvas para cada fração
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, accuracies_1_3, marker='o', color='r', label='1/3 da base')
        plt.plot(epochs_list, accuracies_2_3, marker='o', color='b', label='2/3 da base')
        plt.plot(epochs_list, accuracies_3_3, marker='o', color='g', label='3/3 da base')
        
        # Configurações do gráfico
        plt.title('Épocas x Acurácia x Amostras')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.grid(True)
        plt.xticks(epochs_list)
        
        # Legenda na ponta esquerda da parte inferior
        plt.legend(loc='lower right')
        
        # Salvar e mostrar o gráfico
        graph_path = f'{PLOT_FOLDER}/accuracy_x_epochs_x_samples.png'
        plt.savefig(graph_path, format='png', dpi=300)
        
        if show_plot:
            plt.show()

        return {
            'graph_base64': encode_img_base64(graph_path)
        }
        
    except Exception as e:
        raise RuntimeError(f'Erro ao gerar gráfico: \n{e}\n')