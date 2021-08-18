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

    def createDirectoryFor_GoodBadRawData_MainFile(self):
        try:
            path = os.path.join("Training_Raw_Validated_File/", "Good_Raw_MainFile/")
            if not os.path.isdir(path):
                os.makedirs(path)

            path = os.path.join("Training_Raw_Validated_File/", "Bad_Raw_MainFile/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as ex:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, 'Error while creating MainFile Good and Bad Directory %s' % ex)
            file.close()
            raise OSError

    def createDirectoryFor_GoodBadRawData_AdditionalFile(self):
        try:
            path = os.path.join("Training_Raw_Validated_File/", "Good_Raw_AdditionalFile")
            if not os.path.isdir(path):
                os.makedirs(path)

            path = os.path.join("Training_Raw_Validated_File/", "Bad_Raw_AdditionalFile")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as ex:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, 'Error while creating Additional Good and Bad Directory %s' % ex)
            file.close()
            raise OSError

    def deleteExistingGoodDataTrainingDir_MainFile(self):
        try:
            path = "Training_Raw_Validated_File/"
            if os.path.isdir(path + 'Good_Raw_MainFile/'):
                shutil.rmtree(path + 'Good_Raw_MainFile/')
                file = open('Training_Logs/General_Log.txt', 'a+')
                self.logger.log(file, 'Good Raw Main File Directory deleted Sucessfully !!!')
                file.close()

        except OSError as ex:
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file, 'Error while deleting Main File Good Raw Directory: %s' % ex)
            file.close()
            raise OSError

    def deleteExistingGoodDataTrainingDir_AdditionalFile(self):
        try:
            path = "Training_Raw_Validated_File/"
            if os.path.isdir(path + 'Good_Raw_AdditionalFile/'):
                shutil.rmtree(path + 'Good_Raw_AdditionalFile/')
                file = open('Training_Logs/General_Log.txt', 'a+')
                self.logger.log(file, 'Good Raw Main File Directory deleted Sucessfully !!!')
                file.close()


        except OSError as ex:
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file, 'Error while deleting Good Raw Directory: %s' % ex)
            file.close()
            raise OSError

    def deleteExistingBadDataTrainingDir_MainFile(self):
        try:
            path = "Training_Raw_Validated_File/"
            if os.path.isdir(path + 'Bad_Raw_MainFile/'):
                shutil.rmtree(path + 'Bad_Raw_MainFile/')
                file = open('Training_Logs/General_Log.txt', 'a+')
                self.logger.log(file, 'Bad Raw Additional Directory deleted Sucessfully !!!')
                file.close()

        except OSError as ex:
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file, 'Error while deleting Main File Bad Raw Directory: %s' % ex)
            file.close()
            raise OSError

    def deleteExistingBadDataTrainingDir_AdditionalFile(self):
        try:
            path = "Training_Raw_Validated_File/"
            if os.path.isdir(path + 'Bad_Raw_AdditionalFile/'):
                shutil.rmtree(path + 'Bad_Raw_AdditionalFile/')
                file = open('Training_Logs/General_Log.txt', 'a+')
                self.logger.log(file, 'Bad Raw Additional Directory deleted Sucessfully !!!')
                file.close()
        except OSError as ex:
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file, 'Error while deleting Additional Bad Raw Directory: %s' % ex)
            file.close()
            raise OSError

    def moveBadFilesToArchiveBad_MainFile(self):
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Training_Raw_Validated_File/Bad_Raw_MainFile/'
            if os.path.isdir(source):
                path = 'TrainingArchiveBadData_MainFile'
                if not os.path.isdir(path):
                    os.makedirs(path)
                destination = 'TrainingArchiveBadData_MainFile/Bad_Data_' + str(date) + "_" + str(time)

                if not os.path.isdir(destination):
                    os.makedirs(destination)
                files = os.listdir(source)

                for f in files:
                    if f not in os.listdir(destination):
                        shutil.move(source + f, destination)
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, 'Bad Main files moved to archive')
            path = "Training_Raw_Validated_File"
            if os.path.isdir(path + 'Bad_Raw_MainFile/'):
                shutil.rmtree(path + 'Bad_Raw_MainFile/')
            self.logger.log(file, 'Bad Raw Main Files Data Directory Removed Successfully!!')
            file.close()

        except Exception as e:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while moving bad main files to Archive::%s' % e)
            file.close()
            raise e

    def moveBadFilesToArchiveBad_AdditionalFile(self):
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Training_Raw_Validated_File/Bad_Raw_AdditionalFile/'
            if os.path.isdir(source):
                path = 'TrainingArchiveBadData_AdditionalFile'
                if not os.path.isdir(path):
                    os.makedirs(path)
                destination = 'TrainingArchiveBadData_AdditionalFile/Bad_Data_' + str(date) + "_" + str(time)

                if not os.path.isdir(destination):
                    os.makedirs(destination)
                files = os.listdir(source)

                for f in files:
                    if f not in os.listdir(destination):
                        shutil.move(source + f, destination)
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Bad Additional files moved to archive')
            path = "Training_Raw_Validated_File"
            if os.path.isdir(path + 'Bad_Raw_AdditionalFile/'):
                shutil.rmtree(path + 'Bad_Raw_AdditionalFile/')
            self.logger.log(file, 'Bad Raw Additional Files Data Directory Removed Successfully!!')
            file.close()

        except Exception as e:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, 'Error while moving bad main files to Archive::%s' % e)
            file.close()
            raise e

    def validationFileNameRaw_MainFile(self, mainfile_Regex, main_lengthofdatestampinfile,
                                       main_lengthoftimestampinfile):
        self.deleteExistingBadDataTrainingDir_MainFile()
        self.deleteExistingGoodDataTrainingDir_MainFile()
        self.createDirectoryFor_GoodBadRawData_MainFile()

        onlyfiles = [f for f in os.listdir(self.batch_directory_MainFile)]

        try:
            file = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(mainfile_Regex, filename)):
                    split = re.split('.csv', filename)
                    split = re.split('_', split[0])
                    if len(split[2]) == main_lengthofdatestampinfile:
                        if len(split[3]) == main_lengthoftimestampinfile:
                            shutil.copy("Training_Batch_Files/Main_File/" + filename,
                                        "Training_Raw_Validated_File/Good_Raw_MainFile")
                            self.logger.log(file,
                                            'Valid File Name !! File moved to GoodRaw_Main Directory ::%s' % filename)
                        else:
                            shutil.copy("Training_Batch_Files/Main_File/" + filename,
                                        "Training_Raw_Validated_File/Bad_Raw_MainFile")
                            self.logger.log(file, 'Invalid File Name!! File moved to Bad Raw Main File Directory')
                    else:
                        shutil.copy("Training_Batch_Files/Main_File/" + filename,
                                    "Training_Raw_Validated_File/Bad_Raw_MainFile")
                        self.logger.log(file, 'Invalid File Name!! File moved to Bad Raw Main File Directory')
                else:
                    shutil.copy("Training_Batch_Files/Main_File/" + filename,
                                "Training_Raw_Validated_File/Bad_Raw_MainFile")
                    self.logger.log(file, 'Invalid File Name!! File moved to Bad Raw Main File Directory')
            file.close()
        except Exception as e:
            file = open("Training_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(file, "Error occured while validating Main FileName %s" % e)
            file.close()
            raise e

    def validationFileNameRaw_AdditionalFile(self, additionalfile_Regex, additionalfile_lengthofdatestampinfile,
                                             additionalfile_lengthoftimestampinfile):
        self.deleteExistingBadDataTrainingDir_AdditionalFile()
        self.deleteExistingGoodDataTrainingDir_AdditionalFile()
        self.createDirectoryFor_GoodBadRawData_AdditionalFile()

        onlyfiles = [f for f in os.listdir(self.batch_directory_AdditionalFile)]

        try:
            file = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(additionalfile_Regex, filename)):
                    split = re.split('.csv', filename)
                    split = re.split('_', split[0])
                    if len(split[3]) == additionalfile_lengthofdatestampinfile:
                        if len(split[4]) == additionalfile_lengthoftimestampinfile:
                            shutil.copy("Training_Batch_Files/Additional_File/" + filename,
                                        "Training_Raw_Validated_File/Good_Raw_AdditionalFile")
                            self.logger.log(file,
                                            'Valid File Name !! File moved to GoodRaw_Additional Directory ::%s' % filename)
                        else:
                            shutil.copy("Training_Batch_Files/Additional_File/" + filename,
                                        "Training_Raw_Validated_File/Bad_Raw_AdditionalFile")
                            self.logger.log(file, 'Invalid File Name!! File moved to Bad Raw Additional File Directory')
                    else:
                        shutil.copy("Training_Batch_Files/Additional_File/" + filename,
                                    "Training_Raw_Validated_File/Bad_Raw_AdditionalFile")
                        self.logger.log(file, 'Invalid File Name!! File moved to Bad Raw Additional File Directory')
                else:
                    shutil.copy("Training_Batch_Files/Additional_File/" + filename,
                                "Training_Raw_Validated_File/Bad_Raw_AdditionalFile")
                    self.logger.log(file, 'Invalid File Name!! File moved to Bad Raw Additional File Directory')
            file.close()
        except Exception as e:
            file = open("Training_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(file, "Error occured while validating Additional FileName %s" % e)
            file.close()
            raise e

    def validate_NoOfCol_MainFile(self, noofcol_mainfile):
        try:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            for file in os.listdir('Training_Raw_Validated_File/Good_Raw_MainFile/'):
                csv = pd.read_csv('Training_Raw_Validated_File/Good_Raw_MainFile/' + file)
                if csv.shape[1] == noofcol_mainfile:
                    pass
                else:
                    shutil.move('Training_Raw_Validated_File/Good_Raw_MainFile' + file,
                                'Training_Raw_Validated_File/Bad_Raw_MainFile')
                    self.logger.log(f,
                                    'Invalid Column length for the file !! File moved to bad raw main Directory :: %s' % file)
                self.logger.log(f, 'Main File Columns Length Validated Sucessfully')
            f.close()
        except OSError:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, 'Error Occured while moving file :: %s' % str(OSError))
            f.close()
            raise OSError

        except Exception as e:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e

    def validate_NoOfCol_AdditionalFile(self, noofcol_additionalfile):
        try:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            for file in os.listdir('Training_Raw_Validated_File/Good_Raw_AdditionalFile/'):
                csv = pd.read_csv('Training_Raw_Validated_File/Good_Raw_AdditionalFile/' + file)
                if csv.shape[1] == noofcol_additionalfile:
                    pass
                else:
                    shutil.move('Training_Raw_Validated_File/Good_Raw_AdditionalFile' + file,
                                'Training_Raw_Validated_File/Bad_Raw_AdditionalFile')
                    self.logger.log(f,
                                    'Invalid Column length for the file !! File moved to bad raw additional Directory :: %s' % file)
                self.logger.log(f, 'Additional File Columns Length Validated Sucessfully')
            f.close()
        except OSError:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, 'Error Occured while moving file :: %s' % str(OSError))
            f.close()
            raise OSError

        except Exception as e:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e

def validattion_main(config_path: str, datasource: str) -> None:
    try:
        print("data validation started")
        config = read_params(config_path)
        raw_data = RawDataValidation(config, None)
        # extracting values from prediction schema
        LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = raw_data.values_from_schema()
        # getting the regex defined to validate filename
        regex = raw_data.manualRegexCreation()
        # validating filename of prediction files
        raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
        # validating column length in the file
        raw_data.validateColumnLength(noofcolumns)
        # validating if any column has all values missing
        raw_data.validateMissingValuesInWholeColumn()
        print("data validation completed")
    except Exception as e:
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config", "params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print("started")

    validattion_main(config_path=parsed_args.config, datasource=parsed_args.datasource)
