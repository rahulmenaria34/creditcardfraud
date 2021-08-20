import logging
import os
import argparse
import shutil
import yaml
import json
import subprocess
from src.training.stage_00_Data_Loader import loader_main
from src.training.stage_01_Data_validation import validation_main
from src.training.stage_02_Data_Transform import transform_main
from src.training.stage_03_DB_Operation import db_operation_main
from src.training.stage_04_Cleaning import cleaning_main
from src.training.stage_05_Export import export_main
from src.training.stage_06_train_model import train_main

"""
from wafer.prediction.stage_00_Data_Loader import pred_loader_main
from wafer.prediction.stage_01_Data_validation import pred_validation_main
from wafer.prediction.stage_03_DB_Operation import pred_db_operation_main
from wafer.prediction.stage_04_Cleaning import pred_cleaning_main
from wafer.prediction.stage_05_Export import pred_export_main
from wafer.prediction.stage_02_Data_Transform import pred_transform_main
from wafer.prediction.stage_06_prediction import prediction_main
from webapp.logging_layer.logger.logger import AppLogger
"""
# import platform

# def path_correction(path):
#     if platform.system() == "Windows":
#         path = path.replace("/", r"\\")
#     return path

logger=AppLogger(project_id=1,log_database="training_pipeline",log_collection_name="pipeline")

def begin_training():
    try:
        logger.log("Training begins..")
        args = argparse.ArgumentParser()
        logger.log(f"{args}:args")
        args=dict()
        #args.add_argument("--config", default=os.path.join("config", "params.yaml"))
        #args.add_argument("--datasource", default=None)
        args['config']=os.path.join("config", "params.yaml")
        args['datasource']=None
        parsed_args=args
        logger.log(f"dictionary created.{args}")
        #parsed_args = args.parse_args()
        logger.log(f"{parsed_args}")
        logger.log("Data loading begin..")
        loader_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Data loading completed..")
        logger.log("Data validation began..")
        validation_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Data validation completed..")
        logger.log("Data transformation began..")
        transform_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Data transformation completed..")
        logger.log("Database oberation began..")
        db_operation_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Database oberation completed..")
        logger.log("Cleaning began..")
        cleaning_main(config_path=parsed_args['config'], )
        logger.log("Cleaning completed..")
        logger.log("Export  began..")
        export_main(config_path=parsed_args['config'],)
        logger.log("Export completed..")
        logger.log("Training began")
        train_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Training completed")
    except Exception as e:
        raise e

def begin_prediction():
    try:
        logger.log("Prediction begins..")
        args = argparse.ArgumentParser()
        args['config']=os.path.join("config", "params.yaml")
        args['datasource']=None
        parsed_args=args
        logger.log("Data loading begin..")
        pred_loader_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Data loading completed..")
        logger.log("Data validation began..")
        pred_validation_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Data validation completed..")
        logger.log("Data transformation began..")
        pred_transform_main(config_path=parsed_args['config'],)
        logger.log("Data transformation completed..")
        logger.log("Database oberation began..")
        pred_db_operation_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Database oberation completed..")
        logger.log("Cleaning began..")
        pred_cleaning_main(config_path=parsed_args['config'], )
        logger.log("Cleaning completed..")
        logger.log("Export  began..")
        pred_export_main(config_path=parsed_args['config'], )
        logger.log("Export completed..")
        logger.log("Prediction began")
        prediction_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'])
        logger.log("Prediction completed")
    except Exception as e:
        raise e





def clean():
    dvc_file = "dvc.yaml"
    if os.path.isfile(dvc_file):
        os.remove(dvc_file)
    else:
        print(f"previous {dvc_file} cleaned!")


def train():
    try:
        clean()
        shutil.copy("dvc_pipelines/training/dvc.yaml", "./")
        print("training dvc file copied")
        os.system("dvc repro")
    except Exception as e:
        raise e
    
    


def predict():
    try:
        clean()
        shutil.copy("dvc_pipelines/prediction/dvc.yaml", "./")
        
        os.system("dvc repro")
    except Exception as e:
        raise e


   



def main(training: str) -> None:
    if training == "True":
        print("training")
        train()
    else:
        print("predicting")
        predict()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--training", default=False)
    parsed_args = args.parse_args()
    print("started")
    main(training=parsed_args.training)