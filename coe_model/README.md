# Starting Mlflow Experiment

1. Create your virtual environment
2. Activate virtual environment
3. pip install mlflow
4. Go into your specified folder that you would be running both the tracking server and the MLProject run command
5. mlflow server --host 127.0.0.1 --port 8080
6. cd into the folder with the `MLProject` file
7. mlflow run . --experiment-name COE-Prediction-Race
