from fastapi import FastAPI
from routes import predict, alerts

app = FastAPI(title="Pest AI Backend")

app.include_router(predict.router)
app.include_router(alerts.router)