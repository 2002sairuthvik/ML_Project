import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import joblib
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''This function creates a preprocessor object that applies transformations to the dataset.
        It uses pipelines to handle both numerical and categorical features.
        The numerical features are scaled using StandardScaler after imputing missing values with the median.
        It handles both numerical and categorical columns by applying appropriate imputation and scaling techniques.'''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender','race_ethnicity', 'parental_level_of_education',
                                   'lunch', 'test_preparation_course']
            
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))  # StandardScaler does not center data when with_mean=False
            ])
            
            logging.info("Numerical pipeline created successfully")
            logging.info(" categorical pipeline created successfully")
            
            preprocessor = column_transformer = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            logging.info("Preprocessor object created successfully")
            
        except Exception as e:
            raise CustomException(e, sys)  # Raise custom exception with the error message and sys
        
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")
            
            logging.info("Obtaining preprocessing object")
            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Preprocessing completed successfully")
            # Save the preprocessor object to a file
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
            raise CustomException(e,sys)
        

