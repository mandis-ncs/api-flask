from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import joblib
import pandas as pd
import numpy as np 
from model import predict_from_csv
from utils import create_upload_folder, save_csv, delete_csv

# Instanciar flask
app = FastAPI()


class ResponseItem(BaseModel):
    Case_No: int
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int
    Age_Mons: int
    Sex: str
    Ethnicity: str
    Jaundice: str
    Family_mem_with_ASD: str
    Who_completed_the_test: str
    Class_ASD_Traits: Optional[str]  # Campo opcional

class ResponseModel(BaseModel):
    responses: List[ResponseItem]

# Receber respostas
@app.post('/respostas')
async def upload_data(data: ResponseModel):
    # Cria pasta uploads se nao existir
    folder_name = 'assets/uploads'
    create_upload_folder(folder_name)

    responses = data.responses
    csv_path = os.path.join(folder_name, 'teste.csv')

    save_csv(responses, csv_path)
    return {"status": "received", "data": data}

@app.get('/resultado')
async def send_result():
    folder_name = 'assets/uploads'
    csv_path = os.path.join(folder_name, 'teste.csv')
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail='CSV file not found')    

    categorical_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    label_encoders = {
        col: joblib.load(f'assets/pickles/label_encoder_{col}.pkl') for col in categorical_cols
    }
    onehot_encoder = joblib.load('assets/pickles/onehot_encoder.pkl')
    scaler = joblib.load('assets/pickles/scaler_Age_Mons.pkl')
    model_path = 'assets/db/autism_prediction_model.keras'

    predictions = predict_from_csv(csv_path, model_path, categorical_cols, label_encoders, onehot_encoder, scaler)

    result_value = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]

    # Opcionalmente, excluir o arquivo CSV após a predição
    delete_csv(csv_path)

    return {"resultado": str(result_value)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
