import pandas as pd
import os
import argparse
from src.utility import read_params,get_logger_object_of_training

log_collection="data_getter"

class Data_Getter:
    def __init__(self, config):
        self.config = config
        self.training_file= os.path.join(
            self.config["artifacts"]["training_data"]["Training_FileFromDB"], 
            self.config["artifacts"]["training_data"]["master_csv"])
        self.addition_training_file=os.path.join(
            self.config["artifacts"]["training_data"]["Training_FileFromDB"], 
            self.config["artifacts"]["training_data"]["additional_csv"])

    def get_data(self,logger,is_logging_enable=True):
        try:
            logger.is_log_enable = is_logging_enable
            logger.log(f"Reading training and additional training file " 
            f"from [{self.training_file}] and [{self.addition_training_file}]")
            data= pd.read_csv(self.training_file) # reading the data file
            additional_data=pd.read_csv(self.addition_training_file)
            logger.log("Returning training and additional training file data")
            return data,additional_data
        except Exception as e:
            raise e


def main(config_path: str,is_logging_enable=True) -> None:
    logger = get_logger_object_of_training(config_path=config_path, collection_name=log_collection_name)
    logger.is_log_enable = is_logging_enable
    logger.log("Training begin")
    config = read_params(config_path)
    export = Data_Getter(config)
    data = export.get_data(logger=logger,is_logging_enable=is_logging_enable)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    main(config_path=parsed_args.config)