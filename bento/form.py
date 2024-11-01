import bentoml
import mlflow
import numpy as np

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment(f"/user/test")
# # model_uri can be any URI that refers to an MLflow model
# # Use local path for demostration
# logged_model = 'runs:/36a093c3f1364caebf76d5a7eb7820a8/model'
# bentoml.mlflow.import_model("iris", model_uri=logged_model)


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    bento_model = bentoml.models.get("iris:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        rv = self.model.predict(input_data)
        return np.asarray(rv)