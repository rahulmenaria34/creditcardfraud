from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score, accuracy_score
from urllib.parse import urlparse
import mlflow

class ModelFinder:

    def __init__(self,config):
        self.config = config
        self.config_rf = self.config["training"]["random_forest"] 
        self.config_xgboost = self.config["training"]["xg_boost"]
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def _get_best_params_for_random_forest(self,train_x,train_y):
        try:
            # initializing with different combination of parameters
            param_grid_rf = self.config_rf["param_grid"]
            CV = self.config_rf["cv"]
            verbose = self.config_rf["verbose"]


            #Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.clf, param_grid=param_grid_rf, cv=CV,  verbose=verbose)
            #finding the best parameters
            grid.fit(train_x, train_y)

            mlflow.log_params(grid.best_params_)
            #extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']
            
            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                              max_depth=max_depth, max_features=max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)

            return self.clf
        except Exception as e:
            raise e

    def _get_best_params_for_xgboost(self,train_x,train_y):
        try:
            # initializing with different combination of parameters
            param_grid_xgboost = self.config_xgboost["param_grid"]
            CV = self.config_xgboost["cv"]
            verbose = self.config_xgboost["verbose"]
            # Creating an object of the Grid Search class
            grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid_xgboost, verbose=verbose,cv=CV)
            # finding the best parameters
            grid.fit(train_x, train_y)

            mlflow.log_params(grid.best_params_)
            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            return self.xgb
        except Exception as e:
            raise e


    def get_best_model(self,train_x,train_y,test_x,test_y):
        try:
            # create best model for XGBoost
            with mlflow.start_run(run_name="xg_boost", nested=True):
                xgboost= self._get_best_params_for_xgboost(train_x,train_y)
                prediction_xgboost = xgboost.predict(test_x) # Predictions using the XGBoost Model

                if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    xgboost_score = accuracy_score(test_y, prediction_xgboost)
                    mlflow.log_metric("xgboost_accuracy_score", xgboost_score)
                else:
                    xgboost_score = roc_auc_score(test_y, prediction_xgboost) # AUC for XGBoost
                    mlflow.log_metric("xgboost_roc_auc_score", xgboost_score)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    mlflow.xgboost.log_model(xgboost, "xgboost_model", registered_model_name="xgboost_model")
                else:
                    mlflow.xgboost.log_model(xgboost, "xgboost_model")

            # create best model for Random Forest
            with mlflow.start_run(run_name="random_forest", nested=True):
                random_forest=self._get_best_params_for_random_forest(train_x,train_y)
                prediction_random_forest=random_forest.predict(test_x) # prediction using the Random Forest Algorithm

                if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    random_forest_score = accuracy_score(test_y, prediction_random_forest)
                    mlflow.log_metric("rf_accuracy_score", random_forest_score)
                else:
                    random_forest_score = roc_auc_score(test_y, prediction_random_forest) # AUC for Random Forest
                    mlflow.log_metric("rf_roc_auc_score", random_forest_score)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    mlflow.sklearn.log_model(random_forest, "random_forest_model", registered_model_name="random_forestModel")
                else:
                    mlflow.sklearn.log_model(random_forest, "random_forest_model")

            #comparing the two models
            if(random_forest_score <  xgboost_score):
                return 'XGBoost_best', xgboost
            else:
                return 'RandomForest_best', random_forest

        except Exception as e:
            raise e
