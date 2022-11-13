# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from active_learning.active_learning_api import ActiveLearner
from active_learning.strategies import ActiveLearningStrategies


class RandomSampling(ActiveLearner):
    def __init__(self, max_to_consider=10 ** 6):
        self.max_to_consider = max_to_consider
        self.nb_actions = 0

    def get_strategy(self):
        return ActiveLearningStrategies.RANDOM

    def get_recommended_items_for_labeling(self, workspace_id, model_id, checkpoint_path, dataset_name, category_name, view_dir="", medoids_only=False, sample_size=1):
        if medoids_only:
            res = self.get_unlabeled_medoids(workspace_id, dataset_name, category_name, view_dir)

        else:
           res = self.get_unlabeled_data(workspace_id, dataset_name, category_name, self.max_to_consider)
        
        scores = self.get_per_element_score(res, workspace_id, model_id, checkpoint_path, dataset_name, category_name)
        indices = np.sort(np.argpartition(scores, sample_size)[:sample_size])
        res = np.array(res)[indices]
        return res.tolist()

    def get_per_element_score(self, items, workspace_id, model_id, checkpoint_path, dataset_name, category_name):
        return np.random.rand(len(items))

    
