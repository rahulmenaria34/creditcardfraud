import os
import shutil
import joblib


class File_Operation:
    """
                This class shall be used to save the model after training
                and load the saved model for prediction.

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
    def __init__(self,config):
        # self.file_object = file_object
        # self.logger_object = logger_object
        self.config = config
        self.model_directory = self.config["artifacts"]["models"]["saved_best_models"]["model_dir"]

    def save_model(self,model,cluster_num,fileName):
        """
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None
"""
        try:
            path = os.path.join(self.model_directory,str(cluster_num)) #create seperate directory for each cluster
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path) 

            filePath = os.path.join(path, fileName)
            joblib.dump(model, filePath)
            return 'success'
        except Exception as e:
            raise e

    def load_model(self,filePath):
        """
                    Method Name: load_model
                    Description: load the model file to memory
                    Output: The Model file loaded in memory
                    On Failure: Raise Exception

                    Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None
        """
        try:
            return joblib.load(filePath)
        except Exception as e:
            raise e

    def find_correct_model_file(self,cluster_number):
        """
                            Method Name: find_correct_model_file
                            Description: Select the correct model based on cluster number
                            Output: The Model file
                            On Failure: Raise Exception

                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """
        try:
            list_model_files = list()
            cluster_number = str(cluster_number)
            model_dir_path = os.path.join(self.model_directory, cluster_number)
            model_fileName = os.listdir(model_dir_path)[0]
            model_file_path = os.path.join(model_dir_path, model_fileName)
            return model_file_path
        except Exception as e:
            raise e