from fastapi import APIRouter, HTTPException
from models.ResponseItem import ResponseModel
from models.config import UPLOADS_FOLDER
from service.NeuralNetwork import NeuralNetworkService
from service.utils import create_upload_folder, save_csv, delete_csv
import numpy as np
import os

router = APIRouter()
neural_network_service = NeuralNetworkService()

@router.post('/respostas')
async def upload_data(data: ResponseModel):
    # Cria pasta uploads se nao existir
    create_upload_folder(UPLOADS_FOLDER)

    responses = data.responses
    csv_path = os.path.join(UPLOADS_FOLDER, 'teste.csv')

    save_csv(responses, csv_path)
    return {'status': 'received', 'data': data}


@router.get('/resultado')
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))