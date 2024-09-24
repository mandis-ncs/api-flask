## Análise Assintótica

### DataPreprocessor

#### Função `_preprocess_labels`

```python
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
        raise RuntimeError(f'Erro na codificação das labels da classe alvo: {e}')

```

Essa função itera sobre colunas categóricas para aplicar o `LabelEncoder`

- **Melhor caso**: $O(n)$, onde $n$ é o número de colunas categóricas. O melhor caso ocorre se não houver muitos dados para codificar.

- **Pior caso**: $O(nm)$, onde $n$ é o número de colunas e $m$ o número de elementos em cada coluna.


#### Função `preprocess_categorical`

```python
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
        raise RuntimeError(f'Erro no processamento das variáveis categóricas: {e}')
```

Nesta função, o `OneEncoder` é aplicado a todas as colunas categóricas, o que pode aumentar significativamente o número de coluna.

- **Melhor caso**: $O(nk)$, onde $n$ é o número de colunas categóricas e $k$ o número de valores únicos em cada coluna.

- **Pior caso**: $O(nkm)$, onde $n$ é o número de colunas categóricas, $k$ é o número de valores únicos e $m$ o número de linhas no **dataset**. No pior caso, para cada linha e valor único, gera-se uma nova coluna.

### NeuralNetworkService

#### Função `_create_model`

```python
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

```

A criação do modelo depende das camadas densas e do número de unidades. A complexidade é determinada principalmente pela quantidade de camadas e unidades em cada camada.

- **Melhor caso**: $O(d)$, onde $d$ é o número de camadas e unidades de cada camada.

- **Pior caso**: $O(nd)$, onde $n$ é o número de parâmetros (entradas + unidades) por camada.


#### Função `_cross_validate_model`

```python
def _cross_validate_model(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
    """
    Realiza a validação cruzada no modelo.

    :param X: DataFrame com as caracteristicas.
    :param y: Série com os rótulos.
    :return: Lista de acurácias para cada divisão.
    """
    try:
        # kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # StratifiedKFold mantém a proporção das classes em cada fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in skf.split(X, y):
            model = self._create_model(X.shape[1])
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            accuracies.append(accuracy)

        return accuracies
    
    except Exception as e:
        raise RuntimeError(f'Erro na validação cruzada: {e}')
```

A validação cruzada executa o treinamento múltiplas vezes, dependendo do número de `folds` (divisões dos dados).

- **Melhor caso**: $O(nk)$, onde $n$ é o número de `folds` e $k$ é o custo de treinamento.

- **Pior caso**: $O(nkm)$, onde $m$ é o número de amostras. No pior caso, o número de `epochs` pode aumentar o custo.

#### Conclusão

A análise de complexidade dos principais componentes do **pré-processamento** e da **rede neural** mostra que a maioria das operações depende **linearmente** do tamanho do conjunto de dados. O pior caso ocorre no pré-processamento de colunas categóricas com muitos valores únicos, resultando em um crescimento **quadrático**. No treinamento do modelo, o custo está relacionado ao número de camadas e unidades, mas a validação cruzada pode aumentar **exponencialmente** o tempo de treinamento com um grande número de `folds`.


## Pseudocódigo

```portugol
algoritmo "IA_Treinamento_Modelo_Resumido"
var
    dados : matriz[100][100]
    respostas : vetor[100]
    acuracias : vetor[100]
    // Modelo gerado pela Rede Neural Multilayer Perceptron (MLP)
    modelo : MLP

// Função principal resumida
funcao treinar_e_salvar_modelo(dados : matriz, respostas : vetor) : vetor
inicio
    // 1. Dividir os dados em treino e teste
    dados_treino, dados_teste, respostas_treino, respostas_teste <- dividir_dados(dados, respostas)

    // 2. Normalizar dados
    dados_treino_normalizados <- normalizar_dados(dados_treino)
    dados_teste_normalizados <- normalizar_dados(dados_teste)

    // 3. Criar e treinar modelo
    modelo <- criar_modelo()
    treinar_modelo(modelo, dados_treino_normalizados, respostas_treino)

    // 4. Aplicar a validação cruzada e retornar as acurácias
    acuracias <- validacao_cruzada(modelo, dados_teste_normalizados)

    retorne acuracias
fimfuncao

inicio
    // Execução principal resumida
    predicao <- treino_e_predicao(dados, respostas)
    escreva(": ", predicoes)
fimalgoritmo
```