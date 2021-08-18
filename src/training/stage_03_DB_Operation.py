import os
from wafer.utility import read_params, valuesFromSchemaFunction
import shutil
import sqlite3
import csv
import argparse

class DbOperation:
    def __init__(self, config):
        self.config = config
        self.path = self.config["artifacts"]['training_data']["training_db_dir"]
        self.goodFileDir = self.config["artifacts"]['training_data']['good_validated_raw_dir_updated']
        self.badFileDir = self.config["artifacts"]['training_data']['bad_validated_raw_dir']
        self.DatabaseName = self.config["artifacts"]['training_data']["training_db"]
        self.fileFromDb = self.config["artifacts"]['training_data']["Training_FileFromDB"]
        self.masterCSV = self.config["artifacts"]['training_data']["master_csv"]

    def dataBaseConnection(self):
        try:
            os.makedirs(self.path, exist_ok=True)
            db_path = os.path.join(self.path, self.DatabaseName)
            conn = sqlite3.connect(db_path)
            print("present ??? ", db_path, os.path.isfile(db_path))
        except ConnectionError:
            raise ConnectionError
        return conn

    def createTableDb(self,column_names):
        try:
            conn = self.dataBaseConnection()
            c=conn.cursor()
            c.execute("SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name = 'Good_Raw_Data'")
            if c.fetchone()[0] ==1:
                conn.close()
            else:
                for column_name, dataType in column_names.items():

                    # in try block we check if the table exists, if yes then add columns to the table
                    # else in catch block we will create the table
                    try:
                        #cur = cur.execute("SELECT name FROM {dbName} WHERE type='table' AND name='Good_Raw_Data'".format(dbName=DatabaseName))
                        conn.execute(f'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}')
                    except:
                        conn.execute(f'CREATE TABLE  Good_Raw_Data ({column_name} {dataType})')
                conn.close()

        except Exception as e:
            raise e

    def insertIntoTableGoodData(self):
        conn = self.dataBaseConnection()
        onlyfiles = os.listdir(self.goodFileDir)
        for fileName in onlyfiles:
            try:
                filePath = os.path.join(self.goodFileDir, fileName) 
                with open(filePath, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list_ in (line[1]):
                            try:
                                conn.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=(list_)))
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:

                conn.rollback()
                filePath = os.path.join(self.goodFileDir, fileName) 
                shutil.move(filePath, self.badFileDir)
                conn.close()
                raise e

        conn.close()


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


def db_operation_main(config_path: str, datasource: str) -> None:
    try:
        config = read_params(config_path)
        transform = DbOperation(config)
        schema_path = config['config']['schema_training']
        _,_,column_names,_ = valuesFromSchemaFunction(schema_path)
        transform.createTableDb(column_names)
        transform.insertIntoTableGoodData()
    except Exception as e:
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print("started")
    db_operation_main(config_path=parsed_args.config, datasource=parsed_args.datasource)


