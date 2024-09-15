import os
import sys
from src.logger import logging
from src.exception import custom_exception
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

@dataclass
class data_ingestion_config:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

class initiate_ingestion:
    def __init__(self) -> None:
        self.ingestion_config = data_ingestion_config()

    def intiate_data_ingestion(self):
        logging.info("Data ingestion has started")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info("Train test split initiated")
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise custom_exception(e,sys)
        

if __name__=="__main__":
    obj = initiate_ingestion()
    train_data,test_data = obj.intiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
