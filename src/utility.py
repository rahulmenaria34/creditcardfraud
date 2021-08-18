import json
import yaml
from webapp.logging_layer.logger.logger import AppLogger
import uuid


def get_logger_object_of_training(config_path: str, collection_name) -> AppLogger:
    config = read_params(config_path)
    database_name = config['log_database']['training_database_name']
    execution_id = str(uuid.uuid4())
    logger = AppLogger(project_id=1, log_database=database_name, log_collection_name=collection_name,
                       execution_id=execution_id, executed_by="Avnish Yadav")
    return logger


def read_params(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def values_from_schema_function(schema_path):
    try:
        with open(schema_path, 'r') as r:
            dic = json.load(r)
            r.close()
        main_file = dic['SampleFileName_Main']
        additional_file = dic['SampleFileName_Additional']
        main_length_of_date_stamp_in_file = dic['Main_LengthOfDateStampInFile']
        additional_length_of_date_stamp_in_file = dic['Additional_LengthOfDateStampInFile']
        main_length_of_time_stamp_in_file = dic['Main_LengthOfTimeStampInFile']
        additional_length_of_time_stamp_in_file = dic['Additional_LengthOfTimeStampInFile']
        no_col_main_file = dic['NumberOfColumns_MainFile']
        no_col_additional_file = dic['NumberOfColumns_AdditionalFile']
        main_file_col_name = dic['MainFile_ColName']
        additional_file_col_name = dic['AdditionalFile_ColName']
        return (main_file, additional_file,
                main_length_of_date_stamp_in_file,
                main_length_of_time_stamp_in_file,
                additional_length_of_date_stamp_in_file,
                additional_length_of_time_stamp_in_file,
                main_file_col_name, additional_file_col_name,
                no_col_main_file,
                no_col_additional_file)
    except ValueError:
        raise ValueError

    except KeyError:
        raise KeyError

    except Exception as e:
        raise e
