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

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        


if __name__ == '__main__':
    print()
