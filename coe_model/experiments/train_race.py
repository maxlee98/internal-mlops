import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

# ML Model Creation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# import statsmodels.api as sm
import mlflow

RANDOM_SEED = 42

current_dir = os.path.dirname(os.path.realpath(__file__))
data_fldr = os.path.join(current_dir, "..", "data")
out_dir = os.path.join(current_dir, "..", "data", "output")

coe_df = pd.read_excel(os.path.join(data_fldr, "COE_Export.xlsx"), sheet_name="Yearly")
pp_df = pd.read_excel(os.path.join(data_fldr, "Population.xlsx"), sheet_name="Consolidate")

coe_cat_df = coe_df.loc[coe_df['Category'] == "A", :]
coe_pp = pd.merge(left=coe_cat_df, right=pp_df, left_on="Year", right_on="Year", how="left")
coe_pp_drop = coe_pp.drop(['Year','Category',], axis=1).drop([24], axis=0)

def log_scale(X):
    return np.log1p(X)

# Initialize FunctionTransformer
transformer = FunctionTransformer(log_scale)

X = coe_pp_drop.drop('Value', axis=1)  # Features
y = coe_pp_drop['Value']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_log_scaled = transformer.transform(X_train)
# Add a constant to the X_train_log_scaled for the intercept term
# X_train_log_scaled_with_const = sm.add_constant(X_train_log_scaled)


## Machine Learning Models
### Linear Regression

# Initialize and train your multilinear regression model
model = LinearRegression()
# Fit the linear regression model using statsmodels
model_linear = model.fit(X_train, y_train)
# model_stats = sm.OLS(y_train, X_train_log_scaled_with_const).fit()

### Random Forest Regression
rf = RandomForestRegressor(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["squared_error"],
    'max_leaf_nodes': [30, 50, 100]
}

grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=-1, 
        scoring='neg_mean_squared_error',
        verbose=0
    )

model_forest = grid_forest.fit(X_train, y_train)

mlflow.set_experiment("COE_Prediction-Race")

# Model evelaution metrics
def eval_metrics(actual, pred):
    mse = metrics.mean_squared_error(actual, pred)
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)

    return(mse, mae, r2)


def mlflow_logging(model, X, y, name):
    
     with mlflow.start_run() as run:
        mlflow.set_tracking_uri("http://127.0.0.1:8080/")
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)      

        mlflow.log_param("sample_input", X.sample(1).to_dict(orient='records')[0])

        pred = model.predict(X)

        # Log sample output variable
        mlflow.log_metric("sample_output", y.sample(1).values[0])

        # Metrics from evaluation
        (mse, mae, r2) = eval_metrics(y, pred)

        if name == "LinearRegression":
            model_params = model.get_params()
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            # Log coefficients for Linear Regression
            feature_importance = dict(zip(X.columns, model.coef_))
            mlflow.log_params(feature_importance)

        else:
            # Logging best parameters from gridsearch
            mlflow.log_params(model.best_params_)
            mlflow.log_metric("Mean CV score", model.best_score_)
        
        # Log the metrics
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", mse**0.5)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("R-Square Score", r2)

        # Plot predictions vs actuals
        plt.figure(figsize=(12, 6))
        plt.plot(y.index, y, label='Actual', color='blue')
        plt.plot(y.index, pred, label='Prediction', linestyle='--', color='red')
        plt.title('Predictions vs Actuals')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save the plot as an artifact
        plt.savefig(f'{name}-predictions_vs_actuals.png')
        mlflow.log_artifact(f'{name}-predictions_vs_actuals.png')

        # Logging artifacts and model
        # mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)
        
        mlflow.end_run()

mlflow_logging(model_forest, X_test, y_test, "RandomForestRegressor")
mlflow_logging(model_linear, X_test, y_test, "LinearRegression")