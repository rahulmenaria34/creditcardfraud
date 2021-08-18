import os
import pandas
import argparse
from wafer.utility import read_params, valuesFromSchemaFunction


class DataTransform:
     def __init__(self, config):
          self.goodDataPath = config["artifacts"]['training_data']['good_validated_raw_dir']
          self.goodDataPath_updated = config["artifacts"]['training_data']['good_validated_raw_dir_updated']
          os.makedirs(self.goodDataPath_updated, exist_ok=True)

     def replaceMissingWithNull(self):
          try:
               onlyfiles = os.listdir(self.goodDataPath)
               for fileName in onlyfiles:
                    filePath = os.path.join(self.goodDataPath, fileName)
                    csv = pandas.read_csv( filePath)
                    csv.fillna('NULL',inplace=True)
                    # removing Wafer- Prefix
                    csv['Wafer'] = csv['Wafer'].str[6:]
                    # save the changes to the updated dir
                    updated_filePath = os.path.join(self.goodDataPath_updated, fileName)
                    csv.to_csv(updated_filePath, index=None, header=True)
          except Exception as e:
              raise e


def transform_main(config_path: str, datasource: str) -> None:
    try:
        print("data Transformation started")
        config = read_params(config_path)
        transform = DataTransform(config)
        transform.replaceMissingWithNull()
        print("data Transformation completed")
    except Exception as e:
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print("started")
    transform_main(config_path=parsed_args.config, datasource=parsed_args.datasource)