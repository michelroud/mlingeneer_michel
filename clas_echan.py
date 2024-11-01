import pandas as pd
from prefect import flow,task
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from datetime import timedelta, datetime
from prefect.client.schemas.schedules import IntervalSchedule
from prefect import flow, task, get_run_logger
import mlflow
from sklearn.metrics import mean_absolute_error,mean_squared_error
from mlflow.models import infer_signature
from pathlib import Path
import mlflow.pyfunc
import pickle
#schedule = Schedule(clocks=[CronClock("0 0 * * 0")])  # Tous les dimanches à minuit
#schedule = Schedule(clocks=[CronClock("*/2 * * * *")])



@task
def transformation():
    logger = get_run_logger()
    logger.info("Début de la transformation")
    df_con=pd.DataFrame()
    mil=[]
    LI=pd.ExcelFile("Cours-de-change-2018_2024.xlsx")
    for g in LI.sheet_names:    
        df=pd.read_excel("Cours-de-change-2018_2024.xlsx",sheet_name=g)
        df=df.dropna(axis=0,how='all')
        som = {}  # Initialisez le dictionnaire som sans aucune clé initialement
        cle = ["DEVISES", "MIN EUR", "CMP EUR", "MAX EUR", "MIN USD", "CMP USD", "MAX USD"]

        # Initialisez chaque clé dans cle avec une liste vide dans som
        for c in cle:
            som[c] = []

        for inde, val in df.iterrows():
            bor = None
            for col in df.columns:
                if val[col] in cle:  # Trouver la colonne contenant une valeur dans cle
                    bor = df.columns.get_loc(col) + 1  # Obtenir l'index de la colonne suivante
                    break
            
            if bor is not None:  # Si une valeur dans cle a été trouvée
                som[val[col]].extend(val.iloc[bor:].tolist())  # Ajoutez les valeurs des colonnes suivantes à la liste existante
      
        dt=pd.DataFrame(som)
        mil.append(dt)  
        dt=pd.concat(mil,axis=0)
    dt=dt[dt["DEVISES"].notna()]
    dt["DEVISES"]=pd.to_datetime(dt["DEVISES"])
    logger.info("Fin de la transformation") 
    return dt

@task
def model_pickle(df):
    logger = get_run_logger()
    logger.info("Début de la modélisation")
    df=df[["DEVISES","CMP EUR"]]
    df=df.rename(columns={"CMP EUR":"y","DEVISES":"ds"})
    x=int(len(df.index)*0.8)
    x_train=df.iloc[:x]
    y_test=df.iloc[x:]
    model=Prophet()
    model.fit(x_train)
    app=model.predict(y_test)
    #y_pre=app["yhat"]
    filename = "propht_model.pkl"
    pickle.dump(model, open(filename, "wb"))
    
    return filename,df


@task
def modelisationss(filename, df):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"/user/test")

    # Charger le modèle depuis le fichier
    with open(filename, 'rb') as f:
        loaded_model = pickle.load(f)

    with mlflow.start_run():
        # Créer une signature basée sur les données d'entrée et de sortie
        x_train = df.iloc[:int(len(df.index)*0.8)]
        y_test = df.iloc[int(len(df.index)*0.8):]
        
        signature = infer_signature(x_train)
        reg_model_name = "prophet"

        print("--")
        # Sauvegarder le modèle avec MLflow
        mlflow.log_artifact_local(filename, artifact_path="models/prophets")
        # mlflow.pyfunc.log_model(
        #     python_model=loaded_model,
        #     artifact_path="prophet",
        #     signature=signature,
        #     registered_model_name=reg_model_name
        # )
    
    return signature

    



#         meb=mean_absolute_error(y_test["y"],y_pre)
#         mebr=mean_squared_error(y_test["y"],y_pre)
            
#         mlflow.log_param("n_estimators", 100)
#         mlflow.log_param("max_depth", 5)
#         mlflow.log_metric("mea", meb)
#         mlflow.log_metric("smea", mebr)
#             # Suivre les paramètres du modèle
#         mlflow.log_param("model_type", "prophet")
#             # Enregistrer le modèle
#         #mlflow.prophet.save_model(model, model_uri.resolve())
#         #mlflow.prophet.save_model(model)
#         mlflow.prophet.log_model(model ,"prophet") 
# # Obtenir le lien du modèle enregistré dans le suivi MLflow
#         mlflow.prophet.save_model(model, path="model.pkl")


    # print(f"Modèle enregistré sous : {mlflow.get_artifact_uri('model')}")
    # logger.info("Fin de la modélisation")



#     # Exemple de données pour la prédiction (vous devez ajuster cela selon vos données)
#     # Remplacez `data` par vos propres données sur lesquelles faire des prédictions
#     data = pd.DataFrame({
#         "ds": pd.date_range(start="2024-01-01", periods=10, freq='D')
#     })
    
#     predictions = model.predict(data)
#     logger.info("Prédictions effectuées avec succès.")
#     return predictions

@flow
def tau_echanges():
    df= transformation()
    filename,df=model_pickle(df)
    x=modelisationss(filename,df)
    #predictions = load_model_and_predict()

    #return predictions

tau_echanges()

# #.serve(name="flowing", schedule=IntervalSchedule(interval=timedelta(minutes=10), 
#                                                     anchor_date=datetime(2026, 1, 1, 0, 0), 
#                                                     timezone="America/Chicago"))
