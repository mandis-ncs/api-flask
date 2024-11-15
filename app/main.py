from fastapi import FastAPI
from controller.routes import router
from service.NeuralNetwork import NeuralNetworkService

app = FastAPI()
app.include_router(router)

nn_service = NeuralNetworkService()
nn_service.train_and_save_model()

# uvicorn main:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
