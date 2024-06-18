# Starting Mlflow Experiment

1. Create your virtual environment
2. Activate virtual environment
3. pip install mlflow
4. Go into your specified folder that you would be running both the tracking server and the MLProject run command
5. mlflow server --host 127.0.0.1 --port 8080
6. cd into the folder with the `MLProject` file
7. To use another file for running mlflow
   - Change the MLProject `name` and file called in `command`
   - Change the file python script within to match the same name in MLProject
8. `mlflow run . --experiment-name [EXPERIMENT NAME]`
   - `mlflow run . --experiment-name COE_Prediction-Race`
