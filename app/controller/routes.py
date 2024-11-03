from fastapi import APIRouter, HTTPException, status, Depends
from models.ResponseItem import ResponseItem
from models.ResponseEntity import ResponseEntity
from metrics.evaluate_accuracy import plot_accuracy_by_epochs, evaluate_accuracy_by_sample_size, accuracy_x_epochs_x_samples
from service.NeuralNetwork import NeuralNetworkService
from service.database import get_db, create
from service.utils import read_csv_and_create_objects
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
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

        # Obter o registro da tabela qchat pelo id
        response_entity = db.query(ResponseEntity).filter(ResponseEntity.id == id).first()

        if response_entity is None:
            raise HTTPException(status_code=404, detail=f"Registro com id {id} não encontrado")

        # Atualizar o valor de Class_ASD_Traits com o resultado da previsão
        response_entity.Class_ASD_Traits = str(result_value)
        
        # Confirmar a alteração no banco de dados
        db.commit()
        db.refresh(response_entity)

        return {'resultado': str(result_value)}
    
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
        return evaluate_accuracy_by_sample_size(DATA_PATH, 100, 1000, 100)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")
    
@router.get('/metrics/accuracy_x_epochs_x_samples', status_code=status.HTTP_200_OK)
async def get_metrics():
    try:
        return accuracy_x_epochs_x_samples()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")
    

@router.get('/test')
async def test(db: Session = Depends(get_db)):
    try:
        responses = read_csv_and_create_objects()

        for response in responses:
            id = create(db, response)

            predictions = neural_network_service.predict_from_db(id, db)
            result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]

            # Obter o registro da tabela qchat pelo id
            response_entity = db.query(ResponseEntity).filter(ResponseEntity.id == id).first()

            if response_entity is None:
                raise HTTPException(status_code=404, detail=f"Registro com id {id} não encontrado")

            # Atualizar o valor de Class_ASD_Traits com o resultado da previsão
            response_entity.Class_ASD_Traits = str(result_value)
            
            # Confirmar a alteração no banco de dados
            db.commit()
            db.refresh(response_entity)

        return {'result': 'ok'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")