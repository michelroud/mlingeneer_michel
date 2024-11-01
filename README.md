Start services `docker-compose --env-file config.env up -d --build`

Go to `localhost:9001`
2.11.5
Get MINIO Acces Key and save into `config.env/MINIO_ACCESS_KEY`

Stop services `docker-compose down`

Start again `docker-compose --env-file config.env up -d --build`

https://github.com/renaxtdb/mlflow-prefect-jupyter-docker-compose/blob/main/docker-compose.yaml

https://farisology.com/getting-the-most-out-of-mlops-with-zenml-2
______________________________________________

`docker-compose -f --env-file config.env up -d --build`

prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api








