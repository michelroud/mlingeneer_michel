import mlflow
import bentoml
import os
from fastapi import FastAPI, Depends

import pandas as pd
# # Configurer MLflow pour utiliser Minio
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
# os.environ["AWS_ACCESS_KEY_ID"] = "yopIwOysXBiYbwgi6hWzcJ"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "YxdNUjnqmmmGng2ReySdfse2Pqstu3N6zE8MK79S"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"/user/test")

# # app = FastAPI()

model_url='runs:/2686c9b1a6cf4c46ba86cbf9ce43d257/model'
# # # #model_url="s3://mlflow/3/5794802ab16c4ca48b247997f0b7f31e/artifacts/prophet",

bento_model = bentoml.mlflow.import_model(
    name="prophet_auto",
    model_uri=model_url,
    signatures={"predict": {"batchable": True}},
    labels={"training-set": "data-v1"},
    metadata={"param_a": 0.2}
)


# @bentoml.service()
# @bentoml.mount_asgi_app(app, path="/v1")
# class lassifier:
#     #
#     def __init__(self):
        
#         self.bento_model = bentoml.models.get("prophet_models:latest")
#         # mlflow_model_path = self.bento_model.path_of(bentoml.mlflow.MLFLOW_MODEL_FOLDER)
#         # self.loaded_pytorch_model = mlflow.prophet.load_model(mlflow_model_path)

#     @app.get("/")
#     def hello(self):
#         self.future = self.bento_model.make_future_dataframe(periods=365)
#         self.forecast = self.bento_model.predict(self.future)
        
#         return {"forecast": self.forecast["yhat"].tail().to_dict()}


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class Classifier:
    bento_model = bentoml.models.get("prophet_auto:latest")
    
    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self):
        #self.future = self.model.make_future_dataframe(periods=365)
        last_date = pd.to_datetime('2023-12-31')
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)
        self.forecast = self.model.predict(pd.DataFrame({'ds': future_dates}))
        #rv = self.model.predict(input_data)
        return self.forecast.to_dict()