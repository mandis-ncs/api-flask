from fastapi import APIRouter, HTTPException, status, Depends
from models.ResponseItem import ResponseItem
from models.config import UPLOADS_FOLDER
from service.NeuralNetwork import NeuralNetworkService
from service.utils import create_upload_folder, save_csv, delete_csv
from service.crud import get_db, create
from sqlalchemy.orm import Session
import numpy as np
import os

router = APIRouter()
neural_network_service = NeuralNetworkService()

@router.post('/respostas', status_code=status.HTTP_201_CREATED)
async def upload_data(data: ResponseItem, db: Session = Depends(get_db)):
    try:
        # Cria pasta uploads se nao existir
        id = create(db, data)

        return {'status': 'received', 'qchat_id': id, 'data': data}
    
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erro ao manipular o arquivo: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")


@router.get('/resultado', status_code=status.HTTP_200_OK)
async def send_result():
    csv_path = os.path.join(UPLOADS_FOLDER, 'teste.csv')
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail='CSV file not found')    

    try:
        
        label_encoders, onehot_encoder, scaler = neural_network_service.get_encoders()

        predictions = neural_network_service.predict_from_csv(csv_path, label_encoders, onehot_encoder, scaler)

        result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]
        
        delete_csv(csv_path)

        return {'resultado': str(result_value)}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro nos dados de entrada: {e}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erro ao acessar o arquivo: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {e}")