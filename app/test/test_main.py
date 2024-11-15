from controller.routes import router
from service.NeuralNetwork import NeuralNetworkService
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
import pytest

# Cria a aplicação FastAPI para os testes
app = FastAPI()
app.include_router(router)

nn_service = NeuralNetworkService()
nn_service.train_and_save_model()

@pytest.mark.asyncio
async def test_upload_data():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/respostas', json={
            "A1": 1,
            "A2": 1,
            "A3": 0,
            "A4": 0,
            "A5": 0,
            "A6": 1,
            "A7": 1,
            "A8": 0,
            "A9": 0,
            "A10": 0,
            "Age_Mons": 36,
            "Sex": "m",
            "Ethnicity": "middle eastern",
            "Jaundice": "yes",
            "Family_mem_with_ASD": "no",
            "Class_ASD_Traits": ""
        })

    assert response.status_code == 201
    assert response.json()["status"] == "received"


@pytest.mark.asyncio
async def test_send_result():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        post_response = await client.post('/respostas', json={
            "A1": 1,
            "A2": 1,
            "A3": 0,
            "A4": 0,
            "A5": 0,
            "A6": 1,
            "A7": 1,
            "A8": 0,
            "A9": 0,
            "A10": 0,
            "Age_Mons": 36,
            "Sex": "m",
            "Ethnicity": "middle eastern",
            "Jaundice": "yes",
            "Family_mem_with_ASD": "no",
            "Class_ASD_Traits": ""
        })

        response_json = post_response.json()
        print(f'\nResponse JSON from POST: {response_json}\n')
        id = response_json.get('qchat_id')

        if id is None:
            print(f'Failed to get qchat_id from response: {response_json}')
            pytest.fail("No qchat_id returned in response")

        id = int(id)
        print(f'\nID: {id}\n')

        response = await client.get(f"/resultado/{id}")

        print(f"\nResponse status code: {response.status_code}\n")
        print(f"\nResponse JSON: {response.json()}\n")

    assert response.status_code == 200
    assert "resultado" in response.json()