# Imagem base do Python v3.12.4
# FROM python:3.12.4-slim

# Imagem base do TensorFlow com suporte a GPU
# Executar depois do build: docker run --gpus all -p 8000:8000 --name sofia-fastapi sofia-fastapi:1.0
# FROM tensorflow/tensorflow:latest-gpu

# Imagem base do TensorFlow com suporte a GPU
FROM tensorflow/tensorflow:latest

# Definição do diretório de trabalho do conteiner
WORKDIR /src

# Copia do arquivo da raiz do projeto para o diretório de trabalho
COPY requirements.txt .

# Atualiza o sistema
RUN apt-get update && apt-get install -y python3-venv

# Cria um ambiente virtual
RUN python3 -m venv /venv

RUN /venv/bin/pip install --upgrade pip

# Força a instalação do blinker
RUN pip install --ignore-installed blinker

# Instalação das dependências
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copia do codigo da aplicação para o diretório de trabalho
COPY . .

# Definição do diretório de trabalho da aplicação
WORKDIR /src/app

# Expondo a porta utilizada pela API
EXPOSE 8000

# Comando para executar o servidor Uvicorn com --reload
CMD ["/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t sofia-fastapi:1.0 .
# docker run -p 8000:8000 --name sofia-fastapi sofia-fastapi:1.0