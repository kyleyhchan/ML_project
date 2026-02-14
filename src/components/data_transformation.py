import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer # Use this function for a pipeline
from sklearn.impute import SimpleImputer # Used to replace missing values in a dataset using basic strategies like the mean, median, most frequent value, or a constant.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            
            num_feature = ['writing_score','reading_score']
            cat_feature = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps = [
                    # for this code, we first filling missing value, then standardize it
                    ("imputer", SimpleImputer(strategy = 'median')), # Handling missing value; fill the missing value with median
                    ("scaler", StandardScaler()) # use this for Standardizate the number
                ]
            )
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), # use most_frequent to fill out missing value
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False)) # Usually we don't need to do this
                ]
            )
            logging.info(f"Categorical columns: {cat_feature}")
            logging.info(f"Numerical columns: {num_feature}")
            # Now set the num and cat pipeline to the preprocessor
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_feature),
                ("cat_pipelines",cat_pipeline,cat_feature)
                ]
            )
            return preprocessor # The return a preprocessor to transform you data

        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            # fit_transform: for training data
            # transform: for testing data/ new data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Now we just want to combine features and target
            # np.c_: concatenating arrays along the second axis (columns)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            # Save the pkl file

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)



