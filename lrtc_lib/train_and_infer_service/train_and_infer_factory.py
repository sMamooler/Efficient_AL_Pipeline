# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from  train_and_infer_service.model_type import ModelType, ModelTypes


class TrainAndInferFactory(object):
    def __init__(self):
        self.train_and_infer_nb_instance = None
        self.train_and_infer_random = None
        self.train_and_infer_hf = None
        self.train_and_infer_roberta = None

    def get_train_and_infer(self, model_type: ModelType, checkpoint_path:str):
        if model_type == ModelTypes.NB:
            if self.train_and_infer_nb_instance is None:
                from  train_and_infer_service.train_and_infer_nb import TrainAndInferNB
                self.train_and_infer_nb_instance = TrainAndInferNB()
            return self.train_and_infer_nb_instance
        elif model_type == ModelTypes.RAND:
            if self.train_and_infer_random is None:
                from  train_and_infer_service.train_and_infer_random import TrainAndInferRandom
                self.train_and_infer_random = TrainAndInferRandom()
            return self.train_and_infer_random
        elif model_type == ModelTypes.HFBERT:
            if self.train_and_infer_hf is None:
                from  train_and_infer_service.train_and_infer_hf import TrainAndInferHF
                self.train_and_infer_hf = TrainAndInferHF(50)
            return self.train_and_infer_hf
        elif model_type == ModelTypes.ROBERTA:
            if self.train_and_infer_roberta is None:
                from  train_and_infer_service.train_and_infer_roberta import TrainAndInferRoberta
                self.train_and_infer_roberta = TrainAndInferRoberta(2, checkpoint_path)
            return self.train_and_infer_roberta
        raise Exception(f"model type {model_type.name} is not supported {self.__class__.__name__}")
