import os
import re
import shutil
import pandas as pd
from src.utility import read_params
from webapp.integration_layer.file_management.file_manager import FileManager
from webapp.logging_layer.logger.logger import AppLogger
import argparse
from webapp.logging_layer.logger.logger import AppLogger
from src.utility import get_logger_object_of_training
import time
import uuid

log_collection_name = "data_loader"


def clean_data_source_dir(path, logger=None,is_logging_enable=True):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        for file in os.listdir(path):
            if '.gitignore' in file:
                pass
            logger.log(f"{os.path.join(path, file)}file will be deleted.")
            os.remove(os.path.join(path, file))
            logger.log(f"{os.path.join(path, file)}file has been deleted.")
    except Exception as e:
        raise e


def download_file_from_cloud(cloud_provider, cloud_directory_path,
                             local_system_directory_file_download_path,
                             logger,
                             is_logging_enable=True):
    """
    download_training_file_from_s3_bucket(): It will download file from cloud storage to your system
    ====================================================================================================================
    :param cloud_provider: name of cloud provider amazon,google,microsoft
    :param cloud_directory_path: path of file located at cloud don't include bucket name
    :param local_system_directory_file_download_path: local system path where file has to be downloaded
    ====================================================================================================================
    :return: True if file downloaded else False
    """
    try:

        logger.is_log_enable = is_logging_enable
        file_manager = FileManager(cloud_provider=cloud_provider)
        response = file_manager.list_files(directory_full_path=cloud_directory_path)
        if not response['status']:
            return True
        is_files_downloaded = 1
        for file_name in response['files_list']:
            logger.log(f"{file_name}file will be downloaded in dir--> {local_system_directory_file_download_path}.")
            response = file_manager.download_file(directory_full_path=cloud_directory_path,
                                                  local_system_directory=local_system_directory_file_download_path,
                                                  file_name=file_name)
            is_files_downloaded = is_files_downloaded * int(response['status'])
            logger.log(f"{file_name}file has been downloaded in dir--> {local_system_directory_file_download_path}.")
        return bool(is_files_downloaded)
    except Exception as e:
        raise e


def loader_main(config_path: str, datasource: str,is_logging_enable=True) -> None:
    try:
        logger = get_logger_object_of_training(config_path=config_path, collection_name=log_collection_name)
        logger.is_log_enable = is_logging_enable
        logger.log("Starting data loading operation.\nReading configuration file.")

        config = read_params(config_path)
        cloud_provider = config['cloud_provider']['name']
        cloud_training_batch_file_path = config['data_download']['cloud_training_directory_path']
        cloud_additional_training_file_path = config['additional_data_download'][
            'cloud_additional_training_directory_path']
        local_training_batch_file_path = config['data_source']['Training_Batch_Files']
        local_additional_training_file_path = config['additional_data_source']['additional_training_files']

        logger.log("Configuration detail has been fetched from configuration file.")
        # removing existing training and additional training files from local
        logger.log(f"Cleaning local directory [{local_training_batch_file_path}] "
                   f"and [{local_additional_training_file_path}] used for training.")
        clean_data_source_dir(local_training_batch_file_path)  # removing existing file from local system
        clean_data_source_dir(local_additional_training_file_path)  # removing existing file from local system

        logger.log(f"Cleaning completed. Directory has been cleared now  [{local_training_batch_file_path}]  "
                   f"and [{local_additional_training_file_path}] ")
        # downloading traning and additional training file from cloud into local system
        logger.log("Data will be downloaded from cloud storage into local system")
        download_file_from_cloud(cloud_provider=cloud_provider,
                                 cloud_directory_path=cloud_training_batch_file_path,
                                 local_system_directory_file_download_path=local_training_batch_file_path,
                                 logger=logger,
                                 is_logging_enable=is_logging_enable

                                 )

        download_file_from_cloud(cloud_provider=cloud_provider,
                                 cloud_directory_path=cloud_additional_training_file_path,
                                 local_system_directory_file_download_path=local_additional_training_file_path,
                                 logger = logger,
                                 is_logging_enable = is_logging_enable
                                 )
        logger.log("Data has been downloaded from cloud storage into local system")

    except Exception as e:
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config", "params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print("started")
    loader_main(config_path=parsed_args.config, datasource=parsed_args.datasource)
