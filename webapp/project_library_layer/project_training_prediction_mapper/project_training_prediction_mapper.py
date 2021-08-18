
from wafer.entryPoint import begin_prediction,begin_training
import os
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation


class ExperimentRunner:
    def __init__(self,project_id=None,execution_id=None,executed_by=None):
        self.execution_id=execution_id
        self.executed_by=executed_by
        self.project_id=project_id
        self.mg_db=MongoDBOperation()

        
    def clean_dvc_file_and_log(self):
        try:
            self.mg_db.drop_collection(db_name="training_pipeline",collection_name="pipeline")
            if os.path.exists("dvc.lock"):
                os.remove(os.path.join("dvc.lock"))
            if os.path.exists("dvc.yaml"):
                os.remove(os.path.join("dvc.yaml"))
        except Exception as e:
            raise e
    def start_training(self):
        try:
            self.clean_dvc_file_and_log()
            begin_training()

            return {'status':True,'message':'Training Successfull.'}
        except Exception as e:

            return {'status':False,'message':'Training failed due to .'+str(e)}
        
        
    def start_prediction(self):
        try:

            self.clean_dvc_file_and_log()
            begin_prediction()
            return {'status': True, 'message': 'Prediction Successfull.'}

        except Exception as e:

            return {'status':False,'message':'Prediction failed due to .'+str(e)}
       
    



project_train_and_prediction_mapper = [
    {
        'project_id': 1,
        'ExperimentRunner':ExperimentRunner
    }
]


def get_experiment_class_reference(project_id):
    try:
        for i in project_train_and_prediction_mapper:
            if i['project_id'] == project_id:
                return i['ExperimentRunner']

    except Exception as e:
        raise e
