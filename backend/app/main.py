from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from contextlib import asynccontextmanager

# Set MLflow server tracking uri
mlflow.set_tracking_uri(
    uri="https://dagshub.com/Jorgedelpasado/telecom-churn-prediction-project.mlflow"
)

logged_model = "runs:/3900a46c5cfb4825880093749b01a4b0/telecom_churn"

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model as a PyFuncModel.
    ml_models["xgboost"] = mlflow.pyfunc.load_model(logged_model)
    yield


class ChurnItem(BaseModel):
    call_failure: int
    complains: int
    subscription_length: int
    charge_amount: int
    tariff_plan: int
    age: int
    customer_value: float
    use_per_month: float
    calls_per_month: float
    sms_per_month: float
    dist_nums_per_month: float


app = FastAPI(lifespan=lifespan)


@app.post("/")
def churn_endpoint(item: ChurnItem):
    data = [item.model_dump()]
    churn = ml_models["xgboost"].predict(pd.DataFrame(data))
    return int(churn[0])
