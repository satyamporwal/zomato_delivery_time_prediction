import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import distcalculate

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_datatransformation_object(self):
        try:
            logging.info("data transfromation intiated")

            #setup categorical and numerical data sepreate

            numerical_col = ['Delivery_person_Age', 'Delivery_person_Ratings','Vehicle_condition','distance','multiple_deliveries']
            categorical_col = ['Weather_conditions','Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']

            #Define the custom ranking for each ordinal variable
            Weather_conditions_Map=["Sunny","Stormy","Sandstorms","Windy","Fog","Cloudy"]
            Road_Traffic_Map=["Low","Medium","High","Jam"]
            Type_of_vehicle_map=["bicycle","electric_scooter","scooter","motorcycle"]
            Festival_Map=["No","Yes"]
            City_Map=["Urban","Metropolitian","Semi-Urban"]


            logging.info('pipeline setup')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[

                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[Weather_conditions_Map,Road_Traffic_Map,Type_of_vehicle_map,Festival_Map,City_Map])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_col),
                    ('cat_pipeline',cat_pipeline,categorical_col)
                ]
            )

            return preprocessor
            
            
            logging.info('pipeline completed')
        except Exception as e:
            logging.info('there is error in pipeline')
            raise CustomException(e, sys)
        

    def intiate_data_transformstion(self,train_data,test_data):
        try:
            logging.info('data transformation intiated')
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info('reading data succesfull in train and test path')


            # Calculate the distance between each pair of points
            train_df['distance'] = np.nan
            test_df['distance'] = np.nan

            for i in range(len(train_df)):
                train_df.loc[i, 'distance'] = distcalculate(train_df.loc[i, 'Restaurant_latitude'], 
                                        train_df.loc[i, 'Restaurant_longitude'], 
                                        train_df.loc[i, 'Delivery_location_latitude'], 
                                        train_df.loc[i, 'Delivery_location_longitude'])
            for i in range(len(test_df)):
                test_df.loc[i, 'distance'] = distcalculate(test_df.loc[i, 'Restaurant_latitude'], 
                                        test_df.loc[i, 'Restaurant_longitude'], 
                                        test_df.loc[i, 'Delivery_location_latitude'], 
                                        test_df.loc[i, 'Delivery_location_longitude'])
            logging.info('distance calculation is done')
            logging.info(f'Train Dataframe Head In Logging : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head In Logging : \n{test_df.head().to_string()}')

            #handling null values in dataset
            logging.info('handling null values intited')
            train_df['Delivery_person_Age_imputed'] = train_df['Delivery_person_Age']
            test_df['Delivery_person_Age_imputed'] = test_df['Delivery_person_Age']
            train_df['Delivery_person_Ratings_imputed'] = train_df['Delivery_person_Ratings']
            test_df['Delivery_person_Ratings_imputed'] = test_df['Delivery_person_Ratings']

            train_df['Delivery_person_Age_imputed'][train_df['Delivery_person_Age_imputed'].isnull()] = train_df['Delivery_person_Age'].dropna().sample(train_df['Delivery_person_Age'].isnull().sum()).values
            test_df['Delivery_person_Age_imputed'][test_df['Delivery_person_Age_imputed'].isnull()] = test_df['Delivery_person_Age'].dropna().sample(test_df['Delivery_person_Age'].isnull().sum()).values
            train_df['Delivery_person_Ratings_imputed'][train_df['Delivery_person_Ratings_imputed'].isnull()] = train_df['Delivery_person_Ratings'].dropna().sample(train_df['Delivery_person_Ratings'].isnull().sum()).values
            test_df['Delivery_person_Ratings_imputed'][test_df['Delivery_person_Ratings_imputed'].isnull()] = test_df['Delivery_person_Ratings'].dropna().sample(test_df['Delivery_person_Ratings'].isnull().sum()).values
            
            train_df['Delivery_person_Age_'] = train_df['Delivery_person_Age_imputed']
            test_df['Delivery_person_Age'] = test_df['Delivery_person_Age_imputed']
            train_df['Delivery_person_Ratings'] = train_df['Delivery_person_Ratings_imputed']
            test_df['Delivery_person_Ratings'] = test_df['Delivery_person_Ratings_imputed']


            
            logging.info('handling missing data sucessfull')
            logging.info(f'Train Dataframe Head In Logging : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head In Logging : \n{test_df.head().to_string()}')

            logging.info('getting preprocessor object')
            preprocessing_obj = self.get_datatransformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name,'ID','Delivery_person_ID','Delivery_person_Age_imputed','Delivery_person_Ratings_imputed','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked','Type_of_order']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
                         
                         
                         








