import collections
from math import log
import os
import argparse
from src.utility import get_logger_object_of_training
from src.training.common_utils.file_operations import File_Operation
from sklearn.model_selection import train_test_split
from src.training.common_utils.get_data import Data_Getter
from src.training.common_utils.missing_value_treatment import Preprocessor
from src.training.common_utils.Clustering import KMeansClustering
from src.training.common_utils.best_model_finder import ModelFinder
from src.utility import read_params, valuesFromSchemaFunction
import mlflow
#Creating the common Logging object

log_collection_name="training_model"

class TrainModel:

    def __init__(self, config):
        self.config = config

    def trainingModel(self,logger,is_logging_enable=True):
        try:
            logger.is_log_enable = is_logging_enable
            # Getting the data from the source
            data_getter=Data_Getter(self.config)
            main_data,additional_data=data_getter.get_data(logger=logger,is_logging_enable=is_logging_enable)


            """doing the data preprocessing"""

            preprocessor=Preprocessor(self.config)
            #data=preprocessor.remove_columns(data,['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.

            # create separate features and labels
            #X,Y=preprocessor.separate_label_feature(data,label_column_name='Output')

            # check if missing values are present in the dataset
            is_null_present=preprocessor.is_null_present(main_data,logger,is_logging_enable)

            # if missing values are there, replace them appropriately.
            if(is_null_present):
                X=preprocessor.impute_missing_values(main_data,logger,is_logging_enable) # missing value imputation
            main_data = preprocessor.map_ip_to_country(main_data,additional_data,logger=logger,is_log_enable=is_logging_enable)
            main_data = preprocessor.difference_signup_and_purchase(main_data,logger=logger,is_log_enable=is_logging_enable)
            main_data = preprocessor.encoding_browser(main_data,logger,is_log_enable=True)
            main_data = preprocessor.encoding_source(main_data,logger,is_log_enable=True)
            main_data = preprocessor.encoding_sex(main_data,logger,is_log_enable=True)
            main_data = preprocessor.count_frequency_encoding_country(main_data,logger,is_log_enable=True)
            main_data = preprocessor.remove_unwanted_cols(main_data,logger=logger,return_unwanted_data=False,is_log_enable=True)
            x,y = preprocessor.separate_label_feature(main_data,'class')
            
            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop
            
            with mlflow.start_run(run_name=f"model training", nested=True):
                x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
                # splitting the data into training and test set for each cluster one by one
                model_finder=ModelFinder(self.config,logger,is_logging_enable) # object initialization

                #getting the best model for each of the clusters
                
                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)
                
                file_object = File_Operation(self.config)
                if best_model_name == "RandomForest_best":
                    #mlflow.sklearn.log_model(best_model, best_model_name)
                    # model_dir = self.config["saved_best_models"]["model_dir_RF"]
                    # os.makedirs(model_dir, exist_ok=True)
                    # model_path = self.config["saved_best_models"]["model_path_RF"]
                    # joblib.dump(kmeans, model_path)
                    print(type(best_model))
                    file_object.save_model(best_model,best_model_name + ".joblib")
                else:
                    #mlflow.xgboost.log_model(best_model, best_model_name)
                    # model_dir = self.config["saved_best_models"]["model_dir_XGB"]
                    # os.makedirs(model_dir, exist_ok=True)                        
                    # model_path = self.config["saved_best_models"]["model_path_XGB"]
                    # joblib.dump(kmeans, model_path)
                    file_object.save_model(best_model, best_model_name + ".joblib")

                #saving the best model to the directory.
                # file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                # save_model=file_op.save_model(best_model,best_model_name+str(i))

        except Exception as e:
            raise e



def train_main(config_path: str, datasource: str,is_logging_enable=True) -> None:
    try:
        logger = get_logger_object_of_training(config_path=config_path, collection_name=log_collection_name)
        logger.is_log_enable = is_logging_enable
        logger.log("Training begin")
        config = read_params(config_path)
        # print(config)
        # trainModelObj = TrainModel(config)
        # trainModelObj.trainingModel()
        # mlflow.autolog()
        
        remote_server_uri = config["mlflow_config"]["remote_server_uri"]
        logger.log(f"Remote server uri[{remote_server_uri}]")
        mlflow.set_tracking_uri(remote_server_uri)

        with mlflow.start_run(run_name="main") as active_run:
            trainModelObj = TrainModel(config)
            trainModelObj.trainingModel(logger=logger,is_logging_enable=is_logging_enable)
        print("Training completed")
    except Exception as e:
        raise e
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print(parsed_args.config)
    print(parsed_args.datasource)

    train_main(config_path=parsed_args.config, datasource=parsed_args.datasource)