# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import abc
import logging
import time
from collections import defaultdict
from typing import List
import json

import random
import numpy as np
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

import  data_access.data_access_factory as data_access_factory
import  experiment_runners.experiments_results_handler as res_handler
from  oracle_data_access import oracle_data_access_api
from  active_learning.diversity_calculator import DiversityCalculator
from  active_learning.knn_outlier_calculator import KnnOutlierCalculator
from  active_learning.strategies import ActiveLearningStrategies
from  data_access.core.data_structs import TextElement
from  data_access.data_access_api import DataAccessApi
from  data_access.data_access_factory import get_data_access
from  orchestrator import orchestrator_api
from  orchestrator.orchestrator_api import DeleteModels
from  train_and_infer_service.model_type import ModelType
from  training_set_selector.train_and_dev_set_selector_api import TrainingSetSelectionStrategy
import  oracle_data_access.core.utils as oracle_utils

@dataclass
class ExperimentParams:
    experiment_name: str
    train_dataset_name: str
    dev_dataset_name: str
    test_dataset_name: str
    category_name: str
    workspace_id: str
    model: ModelType
    active_learning_strategies: list
    repeat_id: int
    train_params: dict
    cluster_path: str
    start_with_medoids: bool
    ckpt_path: str



def compute_batch_scores(config, elements):
    data_access = get_data_access()
    unlabeled = data_access.sample_unlabeled_text_elements(config.workspace_id, config.train_dataset_name,
                                                           config.category_name, 10 ** 6)["results"]
    unlabeled_emb = np.array(orchestrator_api.infer(config.workspace_id, config.category_name, config.ckpt_path, unlabeled)["embeddings"])
    batch_emb = np.array(orchestrator_api.infer(config.workspace_id, config.category_name, config.ckpt_path, elements)["embeddings"])

    outlier_calculator = KnnOutlierCalculator(unlabeled_emb)
    outlier_value = outlier_calculator.compute_batch_score(batch_emb)
    representativeness_value = 1 / outlier_value
    diversity_calculator = DiversityCalculator(unlabeled_emb)
    diversity_value = diversity_calculator.compute_batch_score(batch_emb)
    return diversity_value, representativeness_value


class ExperimentRunner(object, metaclass=abc.ABCMeta):
    NO_AL = 'no_active_learning'

    def __init__(self, first_model_positives_num: int, first_model_negatives_num: int,
                 active_learning_suggestions_num: int):
        """
        Init the ExperimentsRunner
        :param first_model_positives_num: the number of positives instances to provide for the first model.
        :param first_model_negatives_num: the number of negative instances to provide for the first model.
        :param active_learning_suggestions_num: the number of instances to be suggested by the active learning strategy
        for the training of the second model.

        """
        self.first_model_positives_num = first_model_positives_num
        self.first_model_negatives_num = first_model_negatives_num
        self.active_learning_suggestions_num = active_learning_suggestions_num
        self.data_access: DataAccessApi = data_access_factory.get_data_access()
        self.cached_first_model_scores = False
        orchestrator_api.set_training_set_selection_strategy(TrainingSetSelectionStrategy.ALL_LABELED)

    def run(self, config: ExperimentParams, active_learning_iterations_num: int, results_file_path: str,
            delete_workspaces: bool = True):

        # key: active learning name, value: list of results oevr iterations (first model has no iterations)
        results_per_active_learning = defaultdict(dict)
        # train first model
        iteration = 0
        
        
        res_dict, pseudo_label_res_dict, first_iter_nb_actions = self.train_first_model(config=config)
        
        res_dict['number of actions'] = first_iter_nb_actions 
        res_handler.save_results(results_file_path, [res_dict])
        res_handler.save_results(results_file_path[:-4]+"_pseudo_labels.csv", [pseudo_label_res_dict])
        results_per_active_learning[self.NO_AL][iteration] = res_dict

        original_workspace_id = config.workspace_id

        for al in config.active_learning_strategies:
            nb_actions = first_iter_nb_actions
               
            orchestrator_api.set_active_learning_strategy(al)
            if not orchestrator_api.is_model_compatible_with_active_learning(al, config.model):
                logging.info(f'skipping active learning strategy {al.name} for model {config.model.name} '
                             f'since the strategy does not support this model.')
                continue

            al_workspace_id = original_workspace_id + "-" + al.name
            if orchestrator_api.workspace_exists(al_workspace_id):
                orchestrator_api.delete_workspace(al_workspace_id, config.ckpt_path)
            orchestrator_api.copy_workspace(original_workspace_id, al_workspace_id)
            config.workspace_id = al_workspace_id
            
            for iteration in range(1, active_learning_iterations_num + 1):
                logging.info(f'Run AL strategy: {al.name}, iteration num: {iteration}, repeat num: {config.repeat_id}\t'
                             f'workspace: {config.workspace_id}')

                res_dict, train_id, pseudo_label_res_dict, iter_nb_actions = self.run_active_learning_iteration(config, al, iteration)
                
                nb_actions += iter_nb_actions
                res_dict['number of actions'] = nb_actions
                res_handler.save_results(results_file_path, [res_dict])
                res_handler.save_results(results_file_path[:-4]+"_pseudo_labels.csv", [pseudo_label_res_dict])
                results_per_active_learning[al.name][iteration] = res_dict

            if delete_workspaces:
                orchestrator_api.delete_workspace(config.workspace_id, config.ckpt_path, DeleteModels.ALL_BUT_FIRST_MODEL)
        if delete_workspaces:
            orchestrator_api.delete_workspace(original_workspace_id, config.ckpt_path)
        return results_per_active_learning

    def train_first_model_cold_start(self, config: ExperimentParams):
        
        # compute and store ucertainty embedding clusters
        elem_2_cluster, cluster_2_elem = orchestrator_api.compute_uncertainty_embeddings(config.workspace_id, config.train_dataset_name, config.category_name, config.model, config.train_params)
        output_dir = "clustering/contract_nli/uncertainty_embedding/"
        
        with open(output_dir+"cluster2elem.json", 'w') as f:
            json.dump(cluster_2_elem, f)
        with open(output_dir+"elem2cluster.json", 'w') as f:
            json.dump(elem_2_cluster, f)


        # find the element with highest AL score
        all_text_elements = orchestrator_api.get_all_text_elements(config.train_dataset_name)
        all_scores = orchestrator_api.get_active_learner_per_element_score(all_text_elements, config.workspace_id, config.ckpt_path, config.category_name)
        gold_labels = oracle_utils.get_gold_labels(config.train_dataset_name, config.category_name)

        highest_score_index = np.argmax(all_scores)
        highest_score_element = all_text_elements[highest_score_index]
        highest_score_uri = highest_score_element.uri
        highest_score_label = gold_labels[highest_score_uri]
        

        # pseduo label its cluster
        highest_score_cluster = elem_2_cluster[highest_score_element]
        highest_score_neighbors = cluster_2_elem[highest_score_cluster]
        neighbors_uris = [e.uri for e in highest_score_neighbors]
        pseudo_labels = [(uri, highest_score_label) for uri in neighbors_uris]
        orchestrator_api.set_labels(config.workspace_id, pseudo_labels)


        logging.info(f'Starting first model training (model: {config.model.name})\tworkspace: {config.workspace_id}')
        new_model_id, new_cosine_similarity_threshold, num_actions, pseudo_labeled_samples = orchestrator_api.train(config.workspace_id, config.category_name, config.model, config.ckpt_path, train_params=config.train_params,
                                                                                                                        first_iteration=True)

        if new_model_id is None:
            raise Exception(f'a new model was not trained\tworkspace: {config.workspace_id}')

        eval_dataset = config.test_dataset_name
        res_dict = self.evaluate(config, al=self.NO_AL, iteration=0, eval_dataset=eval_dataset)
        #res_dict.update(self.generate_al_batch_dict(config))  # ensures AL-related keys are in the results dictionary
        pseudo_label_res_dict = self.generate_pseudo_labels_metrics_dict(config, pseudo_labeled_samples, al=self.NO_AL, iteration=0, eval_dataset=eval_dataset)
        

        logging.info(f'Evaluation on dataset: {eval_dataset}, iteration: 0, first model (id: {new_model_id}) '
                     f'repeat: {config.repeat_id}, is: {res_dict}\t'
                     f'workspace: {config.workspace_id}')

        return res_dict, pseudo_label_res_dict, num_actions

    def run_active_learning_iteration_cold_start(self, config: ExperimentParams, al, iteration):

        # compute and store ucertainty embedding clusters
        elem_2_cluster, cluster_2_elem = orchestrator_api.compute_uncertainty_embeddings(config.workspace_id, config.train_dataset_name, config.category_name, config.model, config.train_params)
      
        # get recommendation 
        # get suggested elements for labeling (and their gold labels)
        suggested_text_elements, suggested_uris_and_gold_labels = \
        self.get_suggested_elements_and_gold_labels(config, al)
        
        suggested_uris_and_psuedo_labels = []
        # find their cluster and speudo label the elements in that cluster
        for index, elem in enumerate(suggested_text_elements):
            cluster = elem_2_cluster[elem.uri]
            cluster_label = suggested_uris_and_gold_labels[index][1]
            new_suggestions_uris = cluster_2_elem[cluster]
            for uri in new_suggestions_uris:
                if (uri!=elem.uri) and ((uri, cluster_label) not in suggested_uris_and_psuedo_labels):
                    suggested_uris_and_psuedo_labels.append((uri, cluster_label)) 

     
        # set gold labels as the user-provided labels of the elements suggested by the active learning strategy
        orchestrator_api.set_labels(config.workspace_id, suggested_uris_and_gold_labels)
        if len(suggested_uris_and_psuedo_labels)>0:
            orchestrator_api.set_labels(config.workspace_id, suggested_uris_and_psuedo_labels)
        

        # train a new model with the additional elements suggested by the active learning strategy
        new_model_id, _, num_actions, _ = orchestrator_api.train(config.workspace_id, config.category_name, config.model, config.ckpt_path, train_params=config.train_params,
                                                first_iteration=False)
        if new_model_id is None:
            raise Exception('New model was not trained')

        # unset pseudo labels after training
        if len(suggested_uris_and_psuedo_labels)>0:
            orchestrator_api.unset_labels(config.workspace_id, config.category_name, list(set([uri for (uri, label) in suggested_uris_and_psuedo_labels])))
            
        # evaluate the new model
        eval_dataset = config.test_dataset_name
        res_dict = self.evaluate(config, al.name, iteration, eval_dataset, suggested_text_elements)
        pseudo_label_res_dict = self.generate_pseudo_labels_metrics_dict(config, suggested_uris_and_psuedo_labels, al.name, iteration, eval_dataset)
        #res_dict.update(al_batch_dict)
        

        logging.info(f'Evaluation on dataset: {eval_dataset}, with AL: {al.name}, iteration: {iteration}, '
                     f'repeat: {config.repeat_id}, model (id: {new_model_id}) is: {res_dict}\t'
                     f'workspace: {config.workspace_id}')
        
        return res_dict, new_model_id, pseudo_label_res_dict, num_actions

        

    def train_first_model(self, config: ExperimentParams):
        if orchestrator_api.workspace_exists(config.workspace_id):
            orchestrator_api.delete_workspace(config.workspace_id, config.ckpt_path)

        orchestrator_api.create_workspace(config.workspace_id, config.train_dataset_name,
                                          dev_dataset_name=config.dev_dataset_name)
        orchestrator_api.create_new_category(config.workspace_id, config.category_name, "No description for you")

        dev_text_elements_uris = orchestrator_api.get_all_text_elements_uris(config.dev_dataset_name)
        dev_text_elements_and_labels = oracle_data_access_api.get_gold_labels(config.dev_dataset_name,
                                                                              dev_text_elements_uris)
        
        if dev_text_elements_and_labels is not None:
            orchestrator_api.set_labels(config.workspace_id, dev_text_elements_and_labels)

        random_seed = sum([ord(c) for c in config.workspace_id])
    
        logging.info(f'random seed: {random_seed}')
        selected_positive_text_elements = self.set_first_model_positives(config, random_seed, config.start_with_medoids)
        selected_negative_text_elements = self.set_first_model_negatives(config, random_seed)
        
        logging.info(f'Starting first model training (model: {config.model.name})\tworkspace: {config.workspace_id}')
        new_model_id, _, num_actions, pseudo_labeled_samples = orchestrator_api.train(config.workspace_id, 
                                                                                      config.category_name, 
                                                                                      config.model, 
                                                                                      config.ckpt_path, 
                                                                                      train_params=config.train_params,
                                                                                      first_iteration=True, 
                                                                                      suggested_text_elements=selected_positive_text_elements 
                                                                                    )
        if new_model_id is None:
            raise Exception(f'a new model was not trained\tworkspace: {config.workspace_id}')

        eval_dataset = config.test_dataset_name
        res_dict = self.evaluate(config, al=self.NO_AL, iteration=0, eval_dataset=eval_dataset)
        #res_dict.update(self.generate_al_batch_dict(config))  # ensures AL-related keys are in the results dictionary
        pseudo_label_res_dict = self.generate_pseudo_labels_metrics_dict(config, pseudo_labeled_samples, al=self.NO_AL, iteration=0, eval_dataset=eval_dataset)
        

        logging.info(f'Evaluation on dataset: {eval_dataset}, iteration: 0, first model (id: {new_model_id}) '
                     f'repeat: {config.repeat_id}, is: {res_dict}\t'
                     f'workspace: {config.workspace_id}')

        return res_dict, pseudo_label_res_dict, num_actions

    def run_active_learning_iteration(self, config: ExperimentParams, al, iteration):
     
        suggested_text_elements, suggested_uris_and_gold_labels = \
        self.get_suggested_elements_and_gold_labels(config, al)
          
        # set gold labels as the user-provided labels of the elements suggested by the active learning strategy
        orchestrator_api.set_labels(config.workspace_id, suggested_uris_and_gold_labels)

        # train a new model with the additional elements suggested by the active learning strategy
        new_model_id, _, num_actions, pseudo_labeled_samples = orchestrator_api.train(config.workspace_id, 
                                                                                      config.category_name, 
                                                                                      config.model, 
                                                                                      config.ckpt_path, 
                                                                                      train_params=config.train_params,
                                                                                      first_iteration=False,
                                                                                      suggested_text_elements=suggested_text_elements)
        if new_model_id is None:
            raise Exception('New model was not trained')

        # evaluate the new model
        eval_dataset = config.test_dataset_name
        res_dict = self.evaluate(config, al.name, iteration, eval_dataset, suggested_text_elements)
        pseudo_label_res_dict = self.generate_pseudo_labels_metrics_dict(config, pseudo_labeled_samples, al.name, iteration, eval_dataset)

        logging.info(f'Evaluation on dataset: {eval_dataset}, with AL: {al.name}, iteration: {iteration}, '
                     f'repeat: {config.repeat_id}, model (id: {new_model_id}) is: {res_dict}\t'
                     f'workspace: {config.workspace_id}')
        
        return res_dict, new_model_id, pseudo_label_res_dict, num_actions

    def generate_pseudo_labels_metrics_dict(self, config, pseudo_labeled_samples, al, iteration, eval_dataset):

        metadata_dict = res_handler.generate_metadata_dict(config, eval_dataset, al, iteration)
        # get gold labels for the pseudo labeled samples
        all_gold_labels = oracle_utils.get_gold_labels(config.train_dataset_name)
        true_labels = {uri:all_gold_labels[uri] for uri, label in pseudo_labeled_samples}
       
        # count tp, tn, fp, fn
        # for fp check the category
        true_positives = [uri for uri, label in pseudo_labeled_samples if label[config.category_name].labels==true_labels[uri][config.category_name].labels==frozenset(["true"])]
        true_negatives = [uri for uri, label in pseudo_labeled_samples if label[config.category_name].labels==true_labels[uri][config.category_name].labels==frozenset(["false"])]
        false_positives = [uri for uri, label in pseudo_labeled_samples if label[config.category_name].labels==frozenset(["true"])!=true_labels[uri][config.category_name].labels]
        false_negatives = [uri for uri, label in pseudo_labeled_samples if label[config.category_name].labels==frozenset(["false"])!=true_labels[uri][config.category_name].labels]

        false_positives_true_labels = [true_labels[uri] for uri in false_positives]
        false_positives_categories = defaultdict(lambda:0)
        for labels in false_positives_true_labels:
            for category, label in labels.items():
                if label.labels==frozenset(["true"]):
                    false_positives_categories[category] = false_positives_categories[category]+1

        pseudo_labels_eval = {'tp':len(true_positives), 'tn': len(true_negatives), 'fp': len(false_positives), 'fn':len(false_negatives)}
        
        pseudo_label_res_dict = {**metadata_dict, **pseudo_labels_eval, **false_positives_categories}
        return pseudo_label_res_dict


    def get_suggested_elements_and_gold_labels(self, config, al):
        start = time.time()
        suggested_text_elements_for_labeling = \
            orchestrator_api.get_elements_to_label(config.workspace_id, config.category_name, config.ckpt_path,
                                                   self.active_learning_suggestions_num)
        end = time.time()
        logging.info(f'{len(suggested_text_elements_for_labeling)} instances '
                     f'suggested by active learning strategy: {al.name} '
                     f'for dataset: {config.train_dataset_name} and category: {config.category_name}.\t'
                     f'runtime: {end - start}\tworkspace: {config.workspace_id}')
        uris_for_labeling = [elem.uri for elem in suggested_text_elements_for_labeling]
        uris_and_gold_labels = oracle_data_access_api.get_gold_labels(config.train_dataset_name, uris_for_labeling,
                                                                      config.category_name)
        return suggested_text_elements_for_labeling, uris_and_gold_labels

    def evaluate(self, config: ExperimentParams, al, iteration, eval_dataset,
                 suggested_text_elements_for_labeling=None):
        metadata_dict = res_handler.generate_metadata_dict(config, eval_dataset, al, iteration)
        labels_counts_dict = res_handler.generate_train_labels_counts_dict(config)
        performance_dict = res_handler.generate_performance_metrics_dict(config, eval_dataset)
        experiment_specific_metrics_dict = \
            self.generate_additional_metrics_dict(config, suggested_text_elements_for_labeling)
        res_dict = {**metadata_dict, **labels_counts_dict, **performance_dict, **experiment_specific_metrics_dict}
        return res_dict

    


    @abc.abstractmethod
    def set_first_model_positives(self, config, random_seed, medoids_only=False) -> List[TextElement]:
        """
        Set the positive instances for the training of the first model.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements and a log message

        """
        func_name = self.set_first_model_positives.__name__
        raise NotImplementedError('users must define ' + func_name + ' to use this base class')

    @abc.abstractmethod
    def set_first_model_negatives(self, config, random_seed) -> List[TextElement]:
        """
        Set the negative instances for the training of the first model.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements and a log message
        """
        func_name = self.set_first_model_negatives.__name__
        raise NotImplementedError('users must define ' + func_name + ' to use this base class')

    @staticmethod
    def generate_al_batch_dict(config, batch_elements=None):
        batch_dict = {}
        model_supports_embeddings = \
            orchestrator_api.is_model_compatible_with_active_learning(ActiveLearningStrategies.DAL, config.model)
        if batch_elements is not None and model_supports_embeddings:
            diversity_value, representativeness_value = compute_batch_scores(config, batch_elements)
            batch_dict["diversity"] = diversity_value
            batch_dict["representativeness"] = representativeness_value
        else:
            batch_dict["diversity"] = "NA"
            batch_dict["representativeness"] = "NA"
        return batch_dict

    def generate_additional_metrics_dict(self, config, suggested_text_elements_for_labeling):
        return {}