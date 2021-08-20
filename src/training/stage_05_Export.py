from _typeshed import Self
import os
from src.utility import read_params, valuesFromSchemaFunction
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import argparse
from src.utility import get_logger_object_of_training
import sqlite3
import csv

log_collection_name="data_export"

class Export:
    def __init__(self, config):
        self.config = config
        self.path = self.config["artifacts"]['training_data']["training_db_dir"]
        self.database_name = self.config["artifacts"]['training_data']["training_db"]
        self.collection_name = self.config["artifacts"]['training_data']["training_collection"]
        self.additional_collection_name=self.config["artifacts"]['training_data']["training_additional_collection"]
        self.file_from_db = self.config["artifacts"]['training_data']["Training_file_from_db"]
        self.additional_file_From_db = self.config["artifacts"]['training_data']["Training_additional_file_drom_db"]
        
        self.masterCSV = self.config["artifacts"]['training_data']["master_csv"]
        self.additionalCSV=self.config["artifacts"]['training_data']["additional_csv"]
        self.mongo_db=MongoDBOperation()
       
    


    def getDataFrameFromDataBase(self,logger,is_log_enable=True):
        try:
            logger.is_log_enable = is_log_enable
            logger.log(f"Creating dataframe of data stored db[{self.database_name}] and collection[{self.collection_name}]")
            df=self.mongo_db.get_dataframe_of_collection(db_name=self.database_name,collection_name=self.collection_name)
            logger.log(f"CSV file has been generated at {os.path.join(self.file_from_db,self.masterCSV)}.")
            df.to_csv(os.path.join(self.file_from_db,self.masterCSV))
            logger.log(f"Creating dataframe of data stored db[{self.database_name}] and collection[{self.additional_collection_name}]")
            df=self.mongo_db.get_dataframe_of_collection(db_name=self.database_name,collection_name=self.additional_collection_name)
            df.to_csv(os.path.join(self.file_from_db,self.additionalCSV))
            logger.log(f"Additional csv file has been generated at {os.path.join(self.file_from_db,self.additionalCSV)}.")
            

        except Exception as e:
            raise e

        


def export_main(config_path: str,is_logging_enable=True) -> None:
    try:
        logger = get_logger_object_of_training(config_path=config_path, collection_name=log_collection_name)
        logger.is_log_enable = is_logging_enable
        config = read_params(config_path)
        export = Export(config)
        logger.log("Generating csv file from data stored in database.")
        export.getDataFrameFromDataBase(is_log_enable=True,logger=logger)
        logger.log("Data has been successfully exported in directory and exiting export pipeline.")
    except Exception as e:
        raise e
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    export_main(config_path=parsed_args.config)