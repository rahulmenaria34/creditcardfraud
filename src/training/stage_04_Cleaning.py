import os
from wafer.utility import read_params, valuesFromSchemaFunction
import argparse
import shutil
import time

class Cleaning:
    def __init__(self,config):
        self.config = config
        self.bad_dest = self.config["artifacts"]['training_data']['bad_validated_raw_dir']
        os.makedirs(self.bad_dest, exist_ok=True)
        print("made")
    
    def deleteExistingBadDataTrainingFolder(self):
        try:
            if os.path.isdir(self.bad_dest):
                shutil.rmtree(self.bad_dest)
        except OSError:
            raise OSError

    def moveBadFilesToArchiveBad(self):
        timeStamp = time.strftime("%Y-%m-%d_%H%M%S")
        try:
            source = self.bad_dest
            if os.path.isdir(source):
                path = self.config["artifacts"]['training_data']["TrainingArchiveBadData"]
                dest = os.path.join(path, f"BadData_{timeStamp}")
                os.makedirs(dest, exist_ok=True)
                print("created")
                files = os.listdir(source)
                for file in files:
                    if file not in os.listdir(dest):
                        source_file_path = os.path.join(source, file)
                        shutil.move(source_file_path, dest)

                shutil.rmtree(source)
        except Exception as e:
            raise e

def cleaning_main(config_path: str) -> None:
    try:
        config = read_params(config_path)
        cleaning = Cleaning(config)
        # cleaning.deleteExistingBadDataTrainingFolder()
        cleaning.moveBadFilesToArchiveBad()
    except Exception as e:
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    cleaning_main(config_path=parsed_args.config)