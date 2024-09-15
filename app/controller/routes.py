from fastapi import APIRouter, HTTPException, status, Depends
from models.ResponseItem import ResponseItem
from metrics.evaluate_accuracy import plot_accuracy_by_epochs, evaluate_accuracy_by_sample_size
from service.NeuralNetwork import NeuralNetworkService
from service.database import get_db, create
from sqlalchemy.orm import Session
import numpy as np
import os

router = APIRouter()
neural_network_service = NeuralNetworkService()

@router.post('/respostas', status_code=status.HTTP_201_CREATED)
async def upload_data(data: ResponseItem, db: Session = Depends(get_db)):
    try:
        id = create(db, data)

        return {'status': 'received', 'qchat_id': id, 'data': data}
    
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erro ao manipular o arquivo: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")


@router.get('/resultado/{id}', status_code=status.HTTP_200_OK)
async def send_result(id: int, db: Session = Depends(get_db)):
    try:
        # Fazer a previsão a partir do banco de dados
        predictions = neural_network_service.predict_from_db(id, db)
        result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]

        return {'resultado': str(result_value)}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro nos dados de entrada: {e}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erro ao acessar o arquivo: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")


@router.get('/metrics/accuracy_by_epochs', status_code=status.HTTP_200_OK)
async def get_metrics():
    try:
        return plot_accuracy_by_epochs()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")
    

@router.get('/metrics/accuracy_by_sample_size', status_code=status.HTTP_200_OK)
async def get_metrics():
    try:
        DATA_PATH = 'assets/db/dataset.csv'
        return evaluate_accuracy_by_sample_size(DATA_PATH, 105, 1050, 105)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")