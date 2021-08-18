import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
# from file_operations import file_methods
import os
import mlflow
import joblib


plt.style.use("fivethirtyeight")


class KMeansClustering:

    def __init__(self, config):
        self.config = config
        self.init = self.config["data_preprocessing"]["KMeansClustering"]["init"]
        self.random_state = self.config["base"]["random_state"]

    def elbow_plot(self,data):
        n_cluster_max = self.config["data_preprocessing"]["KMeansClustering"]["n_cluster_max"]
        artifacts = self.config["artifacts"]["training_data"]["plots"]
        curve = self.config["data_preprocessing"]["KMeansClustering"]["KneeLocator"]["curve"]
        direction = self.config["data_preprocessing"]["KMeansClustering"]["KneeLocator"]["direction"]

        wcss=[] # initializing an empty list
        try:
            for i in range(1,n_cluster_max):
                kmeans=KMeans(n_clusters=i,init=self.init,random_state=self.random_state) # initializing the KMeans object
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,n_cluster_max),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method', fontsize=14)
            plt.xlabel('Number of clusters', fontsize=12)
            plt.ylabel('WCSS', fontsize=12)
            # plt.show()
            os.makedirs(artifacts, exist_ok=True)
            plt.tight_layout()
            #plt.savefig(os.path.join(artifacts,'K-Means_Elbow.PNG'), dpi=100) # saving the elbow plot locally
            
            #mlflow.log_artifact(os.path.join(artifacts,'K-Means_Elbow.PNG'))
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, n_cluster_max), wcss, curve=curve, direction=direction)
            return kn.knee

        except Exception as e:
            raise e


    def create_clusters(self,data,number_of_clusters):
        try:
            mlflow.log_params(self.config["data_preprocessing"]["KMeansClustering"])
            kmeans = KMeans(n_clusters=number_of_clusters, init=self.init, random_state=self.random_state)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            y_kmeans=kmeans.fit_predict(data) #  divide data into clusters

            #mlflow.sklearn.log_model(kmeans, "kmeans")
            model_dir = self.config["artifacts"]["models"]["saved_best_models"]["model_dir_kmeans"]
            os.makedirs(model_dir, exist_ok=True)
            model_path = self.config["artifacts"]["models"]["saved_best_models"]["model_path_kmeans"]
            joblib.dump(kmeans, model_path)
            # self.file_op = file_methods.File_Operation(self.file_object,self.logger_object)
            # self.save_model = self.file_op.save_model(self.kmeans, 'KMeans') # saving the KMeans model to directory
            #                                                                         # passing 'Model' as the functions need three parameters

            data['Cluster']=y_kmeans  # create a new column in dataset for storing the cluster information
            return data

        except Exception as e:
            raise e
