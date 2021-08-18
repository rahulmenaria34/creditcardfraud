import os
from wafer.utility import read_params, valuesFromSchemaFunction
import argparse
import sqlite3
import csv

class Export:
    def __init__(self, config):
        self.config = config
        self.path = self.config["artifacts"]['training_data']["training_db_dir"]
        self.DatabaseName = self.config["artifacts"]['training_data']["training_db"]
        self.fileFromDb = self.config["artifacts"]['training_data']["Training_FileFromDB"]
        self.masterCSV = self.config["artifacts"]['training_data']["master_csv"]

    def dataBaseConnection(self):
        try:
            os.makedirs(self.path, exist_ok=True)
            db_path = os.path.join(self.path, self.DatabaseName)
            conn = sqlite3.connect(db_path)
        except ConnectionError:
            raise ConnectionError
        return conn

    def selectingDatafromtableintocsv(self):
        try:
            conn = self.dataBaseConnection()
            sqlSelect = "SELECT *  FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            #Make the CSV ouput directory
            os.makedirs(self.fileFromDb, exist_ok=True)

            # Open CSV file for writing.
            filePath = os.path.join(self.fileFromDb, self.masterCSV)
            csvFile = csv.writer(open(filePath, 'w', newline=''),delimiter=',', lineterminator='\r\n',quoting=csv.QUOTE_ALL, escapechar='\\')

            # Add the headers and data to the CSV file.
            csvFile.writerow(headers)
            csvFile.writerows(results)

        except Exception as e:
            raise e


def export_main(config_path: str) -> None:
    try:
        config = read_params(config_path)
        export = Export(config)
        export.selectingDatafromtableintocsv()
    except Exception as e:
        raise e
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    export_main(config_path=parsed_args.config)