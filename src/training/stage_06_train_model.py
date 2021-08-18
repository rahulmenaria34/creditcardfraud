import os
import argparse
from wafer.training.common_utils.file_operations import File_Operation
from sklearn.model_selection import train_test_split
from wafer.training.common_utils.get_data import Data_Getter
from wafer.training.common_utils.missing_value_treatment import Preprocessor
from wafer.training.common_utils.Clustering import KMeansClustering
from wafer.training.common_utils.best_model_finder import ModelFinder

from wafer.utility import read_params, valuesFromSchemaFunction
import mlflow
#Creating the common Logging object


class TrainModel:

    def __init__(self, config):
        self.config = config

    def trainingModel(self):
        try:
            # Getting the data from the source
            data_getter=Data_Getter(self.config)
            data=data_getter.get_data()


            """doing the data preprocessing"""

            preprocessor=Preprocessor(self.config)
            data=preprocessor.remove_columns(data,['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.

            # create separate features and labels
            X,Y=preprocessor.separate_label_feature(data,label_column_name='Output')

            # check if missing values are present in the dataset
            is_null_present=preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if(is_null_present):
                X=preprocessor.impute_missing_values(X) # missing value imputation

            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop
            cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(X)

            # drop the columns obtained above
            X=preprocessor.remove_columns(X,cols_to_drop)

            """ Applying the clustering approach"""

            kmeans=KMeansClustering(self.config) # object initialization.
            number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels']=Y

            # getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                with mlflow.start_run(run_name=f"cluster {i}", nested=True):
                    cluster_data=X[X['Cluster']==i] # filter the data for one cluster

                    # Prepare the feature and Label columns
                    cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                    cluster_label= cluster_data['Labels']

                    # splitting the data into training and test set for each cluster one by one
                    x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)


                    model_finder=ModelFinder(self.config) # object initialization

                    #getting the best model for each of the clusters
                    file_object = File_Operation(self.config)
                    best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)
                    if best_model_name == "RandomForest_best":
                        #mlflow.sklearn.log_model(best_model, best_model_name)
                        # model_dir = self.config["saved_best_models"]["model_dir_RF"]
                        # os.makedirs(model_dir, exist_ok=True)
                        # model_path = self.config["saved_best_models"]["model_path_RF"]
                        # joblib.dump(kmeans, model_path)
                        print(type(best_model))
                        file_object.save_model(best_model, i, best_model_name + ".joblib")
                    else:
                        #mlflow.xgboost.log_model(best_model, best_model_name)
                        # model_dir = self.config["saved_best_models"]["model_dir_XGB"]
                        # os.makedirs(model_dir, exist_ok=True)                        
                        # model_path = self.config["saved_best_models"]["model_path_XGB"]
                        # joblib.dump(kmeans, model_path)
                        file_object.save_model(best_model, i, best_model_name + ".joblib")

                    #saving the best model to the directory.
                    # file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                    # save_model=file_op.save_model(best_model,best_model_name+str(i))

        except Exception as e:
            raise e



def train_main(config_path: str, datasource: str) -> None:
    try:
        print("Training begin")
        config = read_params(config_path)
        # print(config)
        # trainModelObj = TrainModel(config)
        # trainModelObj.trainingModel()
        # mlflow.autolog()
        remote_server_uri = config["mlflow_config"]["remote_server_uri"]
        mlflow.set_tracking_uri(remote_server_uri)

        with mlflow.start_run(run_name="main") as active_run:
            trainModelObj = TrainModel(config)
            trainModelObj.trainingModel()
        print("Training completed")
    except Exception as e:
        raise e
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config","params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print(parsed_args.config)
    print(parsed_args.datasource)

    train_main(config_path=parsed_args.config, datasource=parsed_args.datasource)