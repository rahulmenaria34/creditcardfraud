import pandas as pd
import os
import argparse
from wafer.utility import read_params


class Data_Getter:
    def __init__(self, config):
        self.config = config
        self.prediction_file= os.path.join(
            self.config["artifacts"]["prediction_data"]["Prediction_FileFromDB"], 
            self.config["artifacts"]["prediction_data"]["master_csv"])

    def get_data(self):
        try:
            self.data= pd.read_csv(self.prediction_file) # reading the data file
            return self.data
        except Exception:
            raise Exception


def main(config_path: str) -> None:
    config = read_params(config_path)
    export = Data_Getter(config)
    data = export.get_data()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    main(config_path=parsed_args.config)