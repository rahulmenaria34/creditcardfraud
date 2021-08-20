import os
import re
import shutil
import pandas as pd
from src.utility import read_params, values_from_schema_function
from webapp.integration_layer.file_management.file_manager import FileManager
import argparse
import time


class RawDataValidation:
    def __init__(self, config, path=None):
        self.config = config
        if path is None:
            self.Batch_Directory = self.config["data_source"]["Training_Batch_Files"]
        else:
            self.Batch_Directory = path

        self.schema_path = self.config['config']['schema_training']
        self.good_dest = self.config["artifacts"]['training_data']['good_validated_raw_dir']
        self.bad_dest = self.config["artifacts"]['training_data']['bad_validated_raw_dir']

    def values_from_schema(self):
        return values_from_schema_function(self.schema_path)

    def manual_regex_of_training_and_additional_training_file(self):
        training_file_reg_pattern = self.config['reg_pattern_of_file']['training_file']
        additional_training_file_reg_pattern = self.config['reg_pattern_of_file']['additional_training_file']
        return training_file_reg_pattern, additional_training_file_reg_pattern
    

def validation_main(config_path: str, datasource: str) -> None:
    try:
        print("data validation started")
        config = read_params(config_path)
        raw_data = RawDataValidation(config, None)
        # extracting values from prediction schema
        main_file, additional_file,\
        main_length_of_date_stamp_in_file,\
        main_length_of_time_stamp_in_file,\
        additional_length_of_date_stamp_in_file,\
        additional_length_of_time_stamp_in_file,\
        main_file_col_name, additional_file_col_name,\
        no_col_main_file,\
        no_col_additional_file= raw_data.values_from_schema()
        # getting the regex defined to validate filename
        main_regex,additional_regex=raw_data.manual_regex_of_training_and_additional_training_file()

    except Exception as e:
        raise e

    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config", "params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print("started")

    validattion_main(config_path=parsed_args.config, datasource=parsed_args.datasource)
