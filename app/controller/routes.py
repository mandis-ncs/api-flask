from fastapi import APIRouter, HTTPException
from models.ResponseItem import ResponseItem, ResponseModel
from model import predict_from_csv
from utils import create_upload_folder, save_csv, delete_csv
import joblib as jb
import numpy as np
import pandas as pd
import os

router = APIRouter()

UPLOADS_FOLDER = 'assets/uploads'
PKL_FOLDER = 'assets/pickles'
os.makedirs(PKL_FOLDER, exist_ok=True)
DB_FOLDER = 'assets/db'

@router.post('/respostas')
async def upload_data(data: ResponseModel):
    # Cria pasta uploads se nao existir
    create_upload_folder(UPLOADS_FOLDER)

    responses = data.responses
    csv_path = os.path.join(UPLOADS_FOLDER, 'teste.csv')

    save_csv(responses, csv_path)
    return {"status": "received", "data": data}


@router.get('/resultado')
async def send_result():
    csv_path = os.path.join(UPLOADS_FOLDER, 'teste.csv')
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail='CSV file not found')    

    categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    label_encoders = {
        col: jb.load(f'{PKL_FOLDER}/label_encoder_{col}.pkl') for col in categorical_cols
    }
    onehot_encoder = jb.load(f'{PKL_FOLDER}/onehot_encoder.pkl')
    scaler = jb.load(f'{PKL_FOLDER}/scaler_Age_Mons.pkl')
    model_path = f'{DB_FOLDER}/autism_prediction_model.keras'

    predictions = predict_from_csv(csv_path, model_path, categorical_cols, label_encoders, onehot_encoder, scaler)

    result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]

    # Opcionalmente, excluir o arquivo CSV após a predição
    delete_csv(csv_path)

    return {"resultado": str(result_value)}