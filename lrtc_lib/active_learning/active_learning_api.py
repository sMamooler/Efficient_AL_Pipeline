# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0
import os

import abc
import logging
from typing import Sequence
import pandas as pd

from active_learning.strategies import ActiveLearningStrategy
from data_access.core.data_structs import TextElement

from clustering.cluster_statistic import cluster_statistic

class ActiveLearner:

    @abc.abstractmethod
    def get_strategy(self) -> ActiveLearningStrategy:
        raise NotImplementedError("API functions should not be called")

    @abc.abstractmethod
    def get_recommended_items_for_labeling(self, workspace_id: str, model_id: str, checkpoint_path:str, dataset_name: str,
                                           category_name: str, view_dir="", medoids_only=False, sample_size: int = 1) -> Sequence[TextElement]:
        """
        Returns a batch of *sample_size* elements suggested by the active learning module,
        for a given dataset and category, based on the outputs of model *model_id*
        :param workspace_id:
        :param model_id:
        :param dataset_name:
        :param category_name:
        :param sample_size: number of suggested elements to return
        """
        raise NotImplementedError("API functions should not be called")

    def get_per_element_score(self, items: Sequence[TextElement], workspace_id: str, model_id: str, checkpoint_path:str, dataset_name: str,
                              category_name: str) -> Sequence[float]:
        """
        Optional. For a a given sequence of TextElements, return scores per element by the AL module
        :param items:
        :param workspace_id:
        :param model_id:
        :param dataset_name:
        :param category_name:
        """
        raise NotImplementedError("API functions should not be called")

    def get_unlabeled_data(self, workspace_id: str, dataset_name: str, category_name: str, max_to_consider: int) \
            -> Sequence[TextElement]:
        """
        Return a list of up to *max_to_consider* elements that are unlabeled for a given dataset and category.
        :param workspace_id:
        :param dataset_name:
        :param category_name:
        :param max_to_consider:
        """
        from data_access.data_access_factory import get_data_access
        data_access = get_data_access()
        unlabeled = data_access.sample_unlabeled_text_elements(workspace_id, dataset_name, category_name,
                                                               max_to_consider, remove_duplicates=True)["results"]
        logging.info(f"Got {len(unlabeled)} unlabeled elements for active learning")
        return unlabeled
    
    def get_unlabeled_medoids(self, workspace_id: str, dataset_name: str, category_name: str, view_dir: str):
        unlabeled = self.get_unlabeled_data(workspace_id, dataset_name, category_name, self.max_to_consider)

        print("#######################")
        print(f"{len(unlabeled)} unlabeled samples")
        
        view_names = []
        views = []
        view_file = "sbert-distilled-task-adapt_437-clusters.csv"
        #for view_file in os.listdir(view_dir+"clusters/"):
        name = view_file.split("_")[0]
        view_names.append(name)
        view = cluster_statistic(pd.read_csv(view_dir+"clusters/"+view_file), name, view_dir+"stats/")
        views.append(view)

                                                                                                      
        view = views[0]
        # only consider the clusters that are dense enough
        medoids_indices = view.cluster_medoids#[index] for index in range(len(view.dunn_indices))]# if view.dunn_indices[index]>view.dunn_index_threshold]
        medoid_texts = [view.data.loc[index, "text"] for index in medoids_indices]
    

        unlabeled_medoids = [elem for elem in unlabeled if elem.text in medoid_texts]
     

        print("#######################")
        print(f"{len(unlabeled_medoids)} unlabeled medoids")
    

        return unlabeled_medoids
   

