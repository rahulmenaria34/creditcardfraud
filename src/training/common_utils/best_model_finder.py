from joblib import logger
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,roc_curve
from urllib.parse import urlparse
import mlflow

class ModelFinder:

    def __init__(self,config,logger,is_log_enable=True):
        self.config = config
        self.config_rf = self.config["training"]["random_forest"] 
        self.config_xgboost = self.config["training"]["xg_boost"]
        self.rf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.logger=logger
        self.logger.is_log_enable=is_log_enable



    def _get_best_params_for_xgboost(self,train_x,train_y):
        try:
            self.logger.log("Searching best parameter for xgboost")
            # initializing with different combination of parameters
            param_grid_xgboost = self.config_xgboost["param_grid"]
            CV = self.config_xgboost["cv"]
            verbose = self.config_xgboost["verbose"]
            #max_delta_step= [int(x) for x in np.linspace(start=0,stop=9,num=3)]
            # Creating an object of the Grid Search class
            grid= GridSearchCV(estimator=self.xg,param_grid=param_grid_xgboost,
            cv=CV,n_jobs=-1,verbose=verbose)# finding the best parameters
            grid.fit(train_x, train_y)

            mlflow.log_params(grid.best_params_)
            self.logger.log(f"XG boost best parameter[{grid.best_params_}]")
            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']
           
            # creating a new model with the best parameters
            self.logger.log("Creating a new model with XGboost best paramter")
            self.xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger.log("Best XGBoost model has been created.")
            return self.xgb
        except Exception as e:
            raise e



    def _get_best_params_for_random_forest(self,train_x,train_y):
        try:
            self.logger.log("Searching best parameter for RandomForest")
            # initializing with different combination of parameters
            param_grid_rf = self.config_rf["param_grid"]
            CV = self.config_rf["cv"]
            verbose = self.config_rf["verbose"]


            #Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.clf, param_grid=param_grid_rf,
             cv=CV,  verbose=verbose)
            #finding the best parameters
            grid.fit(train_x, train_y)

            mlflow.log_params(grid.best_params_)
            self.logger.log(f"RandomForest best parameter[{grid.best_params_}]")
            #extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']
            
            #creating a new model with the best parameters
            self.logger.log("Creating a new model with XGboost best paramter")
            self.clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                              max_depth=max_depth, max_features=max_features)
            # training the mew model
            
            self.clf.fit(train_x, train_y)
            self.logger.log("Best RandomForest model has been created.")
            return self.clf
        except Exception as e:
            raise e

    def calculate_geometric_mean(self,fpr,tpr,threshold):
        self.gmeans = np.sqrt(tpr * (1 - fpr))
        self.ix = np.argmax(self.gmeans)
        return threshold[self.ix]


    def get_best_model(self,train_x,train_y,test_x,test_y):
        try:
            self.logger.log('Entered the get_best_model of best_model_finder class')
            self.random_forest = self._get_best_params_for_random_forest(train_x,train_y)
            self.prediction_rf = self.random_forest.predict(test_x)
            self.prediction_probab_rf = self.random_forest.predict_proba(test_x)
            self.prediction_probab_rf = self.prediction_probab_rf[:,1]
            self.rf_fpr,self.rf_tpr,self.rf_threshold = roc_curve(test_y,self.prediction_probab_rf)
            self.rf_best_threshold = self.calculate_geometric_mean(self.rf_fpr,self.rf_tpr,self.rf_threshold)
            self.rf_best_threshold = self.rf_best_threshold + 0.05
            self.logger.log('Best Decision Threshold Value for Random Forest is:: %s' %str(self.rf_best_threshold))

            self.prediction_probab_rf = self.prediction_probab_rf >= self.rf_best_threshold.astype('int')
            if (len(test_y.unique()) == 1):
                self.rf_recall = recall_score(test_y,self.prediction_probab_rf)
                self.rf_precision = precision_score(test_y,self.prediction_probab_rf)
                self.rf_f1_score = f1_score(test_y,self.prediction_probab_rf)
                self.logger.log('Recall of Random Forest::' + str(self.rf_recall) +
                    '\t' + 'Precision of Random Forest::' + str(self.rf_precision) +
                    '\t' + 'F1 Score of Random Forest::' + str(self.rf_f1_score))
            else:
                self.rf_roc_auc_score = roc_auc_score(test_y,self.prediction_probab_rf)
                self.rf_recall = recall_score(test_y, self.prediction_probab_rf)
                self.rf_precision = precision_score(test_y, self.prediction_probab_rf)
                self.rf_f1_score = f1_score(test_y,self.prediction_probab_rf)
                self.logger.log('ROC AUC Score of RandomForest ::' + str(self.rf_roc_auc_score)
                                      + '\t' + 'Recall of Random Forest::' + str(self.rf_recall) + '\t' +
                                      'Precision of Random Forest::' + str(self.rf_precision) + '\t' +
                                      'F1-Score of Random Forest ::' + str(self.rf_f1_score))

            self.xgboost = self._get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xg = self.xgboost.predict(test_x)
            self.prediction_probab_xg = self.xg.predict_proba(test_x)
            self.prediction_probab_xg = self.prediction_probab_xg[:, 1]
            self.xg_fpr, self.xg_tpr, self.xg_threshold = roc_curve(test_y, self.prediction_probab_xg)
            self.xg_best_threshold = self.calculate_geometric_mean(self.xg_fpr, self.xg_tpr, self.xg_threshold)
            self.xg_best_threshold = self.xg_best_threshold
            self.logger.log('Best Decision Threshold Value for XGBoost is:: %s' % str(
                self.xg_best_threshold))

            self.prediction_probab_xg = self.prediction_probab_xg >= self.xg_best_threshold
            if (len(test_y.unique()) == 1):
                self.xg_recall = recall_score(test_y, self.prediction_probab_xg)
                self.xg_precision = precision_score(test_y, self.prediction_probab_xg)
                self.xg_f1_score = f1_score(test_y,self.prediction_probab_xg)
                self.logger.log('Recall of XGBoost::' + str(
                    self.xg_recall) + '\t' + 'Precision of XGBoost::' + str(self.xg_precision) + '\t' +
                                      'F1-Score of XGBoost::' + str(self.xg_f1_score))
            else:
                self.xg_roc_auc_score = roc_auc_score(test_y, self.prediction_probab_xg)
                self.xg_recall = recall_score(test_y, self.prediction_probab_xg)
                self.xg_precision = precision_score(test_y, self.prediction_probab_xg)
                self.xg_f1_score = f1_score(test_y, self.prediction_probab_xg)
                self.logger.log('ROC AUC Score of XGBoost ::' + str(self.xg_roc_auc_score) +
                                      '\t' + 'Recall of XGBoost::' + str(self.xg_recall) +
                                      '\t' + 'Precision of XGBoost::' + str(self.xg_precision) +
                                      '\t' + 'F1-Score of XGBoost::' + str(self.xg_f1_score))

            if self.rf_f1_score < self.xg_f1_score:
                #file = open('Training_Logs/Best_Threshold_Value_For_RandomForest.txt','w')
                #self.loggerobject.log(file,str(self.rf_best_threshold))
                return 'XGBoost',self.xgboost
            else:
                #file = open('Training_Logs/Best Threshold_Value_For_XGBoost.txt','w')
                #self.loggerobject.log(file,str(self.xg_best_threshold))
                return 'RandomForest',self.random_forest

        except Exception as e:
            self.logger.log('Exception Occured in get_best_model method of best_model_finder class')
            self.logger.log('Model Selection Failed.Exited the best model finder method of Model Finder Class')
            raise e
