import os 
import sys
from dataclasses import dataclass
from src.utils import evaluate_model

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import custom_exception
from src.utils import save_object
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear Regressor":LinearRegression(),
                "DecisionTree Regressor":DecisionTreeRegressor(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Randomforest Regressor":RandomForestRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "XgBoosting Regressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor()
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            # print("Model trained")

            best_model_score = max(sorted(model_report.values()))
            # print("got the best model score")

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            print(f"best model name:{best_model}")
            if best_model_score<0.6:
                raise custom_exception("There is no best model")
        

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test,y_pred=predicted)
            return r2_score


        except Exception as e:
            raise custom_exception(e,sys)
            



