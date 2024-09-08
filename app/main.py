from fastapi import FastAPI
from controller.routes import router

app = FastAPI()
app.include_router(router)

# uvicorn main:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
