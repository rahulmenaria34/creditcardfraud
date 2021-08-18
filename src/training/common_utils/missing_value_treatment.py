import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
from wafer.utility import read_params
import mlflow
import argparse
from wafer.training.common_utils.get_data import Data_Getter

class Preprocessor:
    def __init__(self, config):
        self.config = config

    def remove_columns(self,data,columns):
        try:
            useful_data= data.drop(labels=columns, axis=1) # drop the labels specified in the columns
            return useful_data
        except Exception as e:
            raise e

    def separate_label_feature(self, data, label_column_name):
        try:
            X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            Y=data[label_column_name] # Filter the Label columns
            return X, Y
        except Exception as e:
            raise e

    def is_null_present(self, data):
        null_present = False
        try:
            null_counts=data.isna().sum() # check for the count of null values per column
            for i in null_counts:
                if i>0:
                    null_present=True
                    break
            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                os.makedirs(
                    self.config["data_preprocessing"]["preprocessed_data_dir"],
                    exist_ok=True
                )
                filePath = os.path.join(
                    self.config["data_preprocessing"]["preprocessed_data_dir"],
                    self.config["data_preprocessing"]["null_values_csv"])
                dataframe_with_null.to_csv(filePath) # storing the null column information to file
            return null_present
        except Exception as e:
            raise e

    def impute_missing_values(self, data):
        try:
            KNNImputer_config = self.config["data_preprocessing"]["KNNImputer"]
            if KNNImputer_config["missing_values"] == "nan":
                KNNImputer_config["missing_values"] = np.nan

            
            mlflow.log_params(KNNImputer_config)

            imputer=KNNImputer(
                n_neighbors=KNNImputer_config["n_neighbors"], 
                weights=KNNImputer_config["weights"],
                missing_values=KNNImputer_config["missing_values"])
            new_array=imputer.fit_transform(data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            new_data=pd.DataFrame(data=new_array, columns=data.columns)
            return new_data
        except Exception as e:
            raise e

    def get_columns_with_zero_std_deviation(self, data):
        columns=data.columns
        data_n = data.describe()
        col_to_drop=[]
        try:
            for x in columns:
                if (data_n[x]['std'] == 0): # check if standard deviation is zero
                    col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            return col_to_drop
        except Exception as e:
            raise e

def preprocess_data(config_path):
    print("This is ",config_path)
    config = read_params(config_path)
    preprocess = Preprocessor(config)
    data = Data_Getter(config).get_data()
    data = preprocess.remove_columns(data, ["Wafer"])
    # create separate features and labels
    X,Y=preprocess.separate_label_feature(data,label_column_name='Output')

    # check if missing values are present in the dataset
    is_null_present=preprocess.is_null_present(X)

    # if missing values are there, replace them appropriately.
    if(is_null_present):
        X=preprocess.impute_missing_values(X) # missing value imputation

    # check further which columns do not contribute to predictions
    # if the standard deviation for a column is zero, it means that the column has constant values
    # and they are giving the same output both for good and bad sensors
    # prepare the list of such columns to drop
    cols_to_drop=preprocess.get_columns_with_zero_std_deviation(X)

    # drop the columns obtained above
    X=preprocess.remove_columns(X,cols_to_drop)

    return X, Y

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    X, Y = preprocess_data(config_path=parsed_args.config)