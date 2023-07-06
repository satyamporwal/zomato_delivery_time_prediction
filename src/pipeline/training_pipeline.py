import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

if __name__=='__main__':
    
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.intiate_data_transformstion(train_data_path,test_data_path)
    modeltrainingconfig = ModelTrainer()
    modeltrainingconfig.initiate_model_training(train_arr,test_arr)