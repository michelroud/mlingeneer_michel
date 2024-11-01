import streamlit as st
import bentoml
import pandas as pd




st.write("Prévision sur 365 jours à partir de")
    # Créer une instance du service BentoML
bento_model = bentoml.models.get("prophet_models:latest")
model = bentoml.mlflow.load_model(bento_model)
    # Utiliser le service pour faire des prévisions
last_date = pd.to_datetime('2023-12-31')
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)
forecast = model.predict(pd.DataFrame({'ds': future_dates}))

# Afficher les résultats dans Streamlit
st.title(forecast['yhat'])
