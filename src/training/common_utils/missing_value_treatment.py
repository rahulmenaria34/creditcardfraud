import pandas as pd
import numpy as np
import os
from src.utility import read_params
import mlflow
import argparse
from src.training.common_utils.get_data import Data_Getter
from rangetree import RangeTree
from collections import Counter
from imblearn.over_sampling import SMOTE
import math
from sklearn.impute import KNNImputer
from datetime import datetime


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

    def is_null_present(self, data,logger,is_log_enable=True):
        try:
            logger.log("Checking null values in dataset")
            logger.is_log_enable=is_log_enable
            null_present = False
            null_counts=data.isna().sum() # check for the count of null values per column
            for i in null_counts:
                if i>0:
                    null_present=True
                    break
            if(null_present): # write the logs to see which columns have null values
                logger.log("Null value found.")
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
                logger.log(f"Null value information has been saved into [{filePath}]")
            else:
                logger.log("Null value not found in dataset.")
            return null_present
        except Exception as e:
            raise e
    
    def remove_unwanted_cols(self,data,logger,return_unwanted_data=False,is_log_enable=True):
        try:
            logger.log("Removing Unwanted Columns Started !!")
            logger.is_log_enable=is_log_enable
            self.df = data
            self.data = self.df.drop(['user_id','signup_time','purchase_time','device_id','source',
                                         'browser','sex','ip_address','IP_Address','Country','d_month','d_year'],axis=1)

            if return_unwanted_data == True:
                self.unwanted_data = self.df[['user_id','signup_time','purchase_time','device_id','source',
                                         'browser','sex','ip_address','Country']]

            logger.log(self.file_object,'Unwanted Columns Deleted Successfully !!')
            #self.logger_object.log(self.file_object,'Sample Feature with 1 Row')
            #self.logger_object.log(self.file_object,str(self.data.head(1)))

            if return_unwanted_data == True:
                return self.data,self.unwanted_data
            else:
                return self.data

        except Exception as e:
            logger.log(self.file_object,'Error occured while removing unwanted columns :: %s' %str(e))
            raise e

    def count_frequency_encoding_country(self,data,logger,is_log_enable=True):
        try:
            self.data=data
            logger.log("Count Frequency Encoding of Country Feature Started")
            logger.is_log_enable=is_log_enable
            country_map = self.data['Country'].value_counts().to_dict()
            self.data['country_encode'] = self.data['Country'].map(country_map)
            logger.log(self.file_object,'Count Frequency Encoding of Country Feature Successfully Completed')
            return self.data
        except Exception as e:
            logger.log('Error while performing Count Frequency Encoding over Country feature:: %s' % str(e))
            raise e


    def encoding_sex(self,data,logger,is_log_enable=True):
        try:
            logger.log("Entered to perform One-Hot Encoding on Sex Feature")
            logger.is_log_enable=is_log_enable
            self.data=data
            sex_df = pd.get_dummies(self.data['sex'],drop_first=True)
            self.data = pd.concat([self.data,sex_df],axis=1)
            logger.log('One-Hot Encoding of Sex Feature Successfully Completed')
            return self.data
        except Exception as e:
            logger.log(self.file_object,'Error while performing One-Hot Encoding over Sex feature:: %s' % str(e))
            raise e



    def encoding_source(self,data,logger,is_log_enable=True):
        try:
            self.data=data
            logger.log("Entered to perform One-Hot Encoding on Source Feature'")
            logger.is_log_enable=is_log_enable
            source_df = pd.get_dummies(self.data['source'],drop_first=True)
            self.data = pd.concat([self.data,source_df],axis=1)
            logger.log('One-Hot Encoding of Source Feature Successfully Completed')
            return self.data
        except Exception as e:
            logger.log('Error while performing One-Hot Encoding over Source feature:: %s' % str(e))
            raise e
    

    def encoding_browser(self,data,logger,is_log_enable=True):
        try:
            logger.log("Checking null values in dataset")
            logger.is_log_enable=is_log_enable
            logger.log('Entered to perform One-Hot Encoding on Browser Feature')
            self.data = data
            browser_df = pd.get_dummies(self.data['browser'],drop_first=True)
            self.data = pd.concat([self.data,browser_df],axis=1)
            logger.log(self.file_object,'One-Hot Encoding of Browser Feature Successfully Completed')
            return self.data
        except Exception as e:
            raise e

    def difference_signup_and_purchase(self,data,logger,is_log_enable=True):
        try:
            logger.log("Checking null values in dataset")
            logger.is_log_enable=is_log_enable
            logger.log('Entered to Finding Difference time between SignUp Time and Purchase Time')
            self.data = data
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])

            signup_df = pd.DataFrame()
            purchase_df = pd.DataFrame()
            tym_difference_df = pd.DataFrame()

            signup_df['s_day'] = self.data['signup_time'].dt.day
            signup_df['s_month'] = self.data['signup_time'].dt.month
            signup_df['s_year'] = self.data['signup_time'].dt.year
            signup_df['s_hour'] = self.data['signup_time'].dt.hour
            signup_df['s_minute'] = self.data['signup_time'].dt.minute
            signup_df['s_seconds'] = self.data['signup_time'].dt.second

            purchase_df['p_day'] = self.data['purchase_time'].dt.day
            purchase_df['p_month'] = self.data['purchase_time'].dt.month
            purchase_df['p_year'] = self.data['purchase_time'].dt.year
            purchase_df['p_hour'] = self.data['purchase_time'].dt.hour
            purchase_df['p_minute'] = self.data['purchase_time'].dt.minute
            purchase_df['p_seconds'] = self.data['purchase_time'].dt.second

            tym_difference_df['d_day'] = purchase_df['p_day'] - signup_df['s_day']
            tym_difference_df['d_month'] = purchase_df['p_month'] - signup_df['s_month']
            tym_difference_df['d_year'] = purchase_df['p_year'] - signup_df['s_year']
            tym_difference_df['d_hour'] = purchase_df['p_hour'] - signup_df['s_hour']
            tym_difference_df['d_minutes'] = purchase_df['p_minute'] - signup_df['s_minute']
            tym_difference_df['d_seconds'] = purchase_df['p_seconds'] - signup_df['s_seconds']

            tym_difference_df['d_day'] = tym_difference_df['d_day'].apply(lambda x: abs(x))
            tym_difference_df['d_month'] = tym_difference_df['d_month'].apply(lambda x: abs(x))
            tym_difference_df['d_year'] = tym_difference_df['d_year'].apply(lambda x: abs(x))
            tym_difference_df['d_minutes'] = tym_difference_df['d_minutes'].apply(lambda x: abs(x))
            tym_difference_df['d_seconds'] = tym_difference_df['d_seconds'].apply(lambda x: abs(x))
            tym_difference_df['d_hour'] = tym_difference_df['d_hour'].apply(lambda x: abs(x))

            self.data = pd.concat([self.data, tym_difference_df], axis=1)
            logger.log('Finding Difference between Signup and Purchase Time Completed Successfully !!')
            return self.data
        except Exception as e:
            raise e
    def map_ip_to_country(self,main_data,add_data,logger,is_log_enable=True):
        try:
            logger.log("Mapping ip to country")
            logger.is_log_enable=is_log_enable
            logger.log('Entered to Map Ip Address to Corresponding Country')
            main_data['IP_Address'] = main_data['ip_address'].apply(lambda x:math.floor(x))
            add_data['Lower_IP'] = add_data['lower_bound_ip_address'].apply(lambda x:math.floor(x))
            add_data['Upper_IP'] = add_data['upper_bound_ip_address'].apply(lambda x:math.floor(x))
            rt = RangeTree()
            for lower,upper,country in zip(add_data['Lower_IP'],add_data['Upper_IP'],add_data['country']):
                rt[lower:upper] = country
            countries = []
            current_tym = datetime.now()
            for ip in main_data['IP_Address']:
                try:
                    countries.append(rt[ip])
                except:
                    countries.append('No Country Found')
            execution_tym = datetime.now() - current_tym
            main_data['Country'] = countries
            logger.log('Mapping of IP Address to Corresponding Country Successfull in %s' % execution_tym)
            return main_data
        except Exception as e:
            raise e

    def impute_missing_values(self, data,logger,is_logging_enable):
        try:
            logger.log("Imputing missing value to replace null values")
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
            logger.log("Imputation of null value has been successfully completed.")
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