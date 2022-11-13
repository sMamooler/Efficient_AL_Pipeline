# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0
import os
import datetime
import logging
from collections import defaultdict
import random 
from typing import List
import tensorflow as tf
import configargparse

import  experiment_runners.experiments_results_handler as res_handler
from  experiment_runners.experiment_runner import ExperimentRunner, ExperimentParams
from  oracle_data_access import oracle_data_access_api
from  active_learning.strategies import ActiveLearningStrategies
from  data_access.core.data_structs import Label, TextElement
from  orchestrator import orchestrator_api
from  orchestrator.orchestrator_api import LABEL_NEGATIVE
from  train_and_infer_service.model_type import ModelTypes


class ExperimentRunnerImbalanced(ExperimentRunner):
    """
    An experiment over imbalanced data.

    The positive instances for the first model are sampled randomly from the true positive instances.
    The negative instances for the first model are sampled randomly from all other instances, and are set as negatives
    (regardless of their gold label).
    """

    def __init__(self, first_model_positives_num: int, first_model_negatives_num: int,
                 active_learning_suggestions_num: int):
        """
        Init the ExperimentsRunner
        :param first_model_positives_num: the number of positive instances to provide for the first model.
        :param first_model_negatives_num: the number of negative instances to provide for the first model.
        :param active_learning_suggestions_num: the number of instances to be suggested by the active learning strategy
        for each iteration (for training the second model and onwards).
        """
        super().__init__(first_model_positives_num, first_model_negatives_num, active_learning_suggestions_num)

    def set_first_model_positives(self, config, random_seed, medoids_only=False) -> List[TextElement]:
        """
        Randomly choose true positive instances.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements
        """
        all_positives = oracle_data_access_api.sample_positives(config.train_dataset_name, config.category_name, 10**6,
                                                                random_seed, medoids_only=medoids_only, cluster_path=config.cluster_path)
       
        
        all_without_duplicates = self.data_access.sample_text_elements(config.train_dataset_name, 10**6,
                                                                       remove_duplicates=True)['results']
      
        
        uris_without_dups = [element.uri for element in all_without_duplicates]
        pos_without_dups = [(uri, label) for uri, label in all_positives if uri in uris_without_dups]
        pos_uris_withour_dups = [uri for uri, label in pos_without_dups]
        positive_elements_without_duplicate = [element for element in all_without_duplicates if element.uri in pos_uris_withour_dups]
        random.seed(random_seed)
        selected_positives = random.choices(pos_without_dups, k=min(self.first_model_positives_num, len(pos_without_dups)))
        selected_positive_elements = random.choices(positive_elements_without_duplicate, k=min(self.first_model_positives_num, len(pos_without_dups)))
        orchestrator_api.set_labels(config.workspace_id, selected_positives)

        logging.info(f'set the label of {len(selected_positives)} true positive instances as positives '
                     f'for category {config.category_name}')
        return selected_positive_elements

    def set_first_model_negatives(self, config, random_seed) -> List[TextElement]:
        """
         Randomly choose from all unlabeled instances.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements
        """
        sampled_unlabeled_text_elements = \
            self.data_access.sample_unlabeled_text_elements(workspace_id=config.workspace_id,
                                                            dataset_name=config.train_dataset_name,
                                                            category_name=config.category_name,
                                                            sample_size=self.first_model_negatives_num,
                                                            remove_duplicates=True)['results']
        negative_uris_and_label = [(x.uri, {config.category_name: Label(LABEL_NEGATIVE, {})})
                                   for x in sampled_unlabeled_text_elements]
        orchestrator_api.set_labels(config.workspace_id, negative_uris_and_label)

        negative_uris = [x.uri for x in sampled_unlabeled_text_elements]
        logging.info(f'set the label of {len(negative_uris_and_label)} random unlabeled instances as negatives '
                     f'for category {config.category_name}')
        return negative_uris

def config_parser():

    
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--num_experiment_repeats", type=int, default=1, 
                        help='number of time to rpeat the experiment')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--ckpt_path", type=str, 
                        help='where the inital model weights are stored')
    parser.add_argument("--dataset", type=str, 
                        help='which dataset to use, contract_nli or opp')
    parser.add_argument("--categories", type=str, 
                        help='categories to use; expected format is cat1,cat2,cat3,...')
    parser.add_argument("--start_with_medoids", type=bool, default=False, 
                        help='if set medoids will be used for the first iteration')
    parser.add_argument("--active_learning_suggestions_num", type=int, default=10, 
                        help='number of samples to label at each iteration')
    parser.add_argument("--cluster_path", type=str, default="", 
                        help='the directiry where clusters are stored')               
    return parser
     
if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # define experiments parameters
    
    experiment_name = args.expname
    active_learning_iterations_num = 5
    num_experiment_repeats = args.num_experiment_repeats
    # for full list of datasets and categories available run: python -m  data_access.loaded_datasets_info
    categories = args.categories.split(",")
    datasets_and_categories = {args.dataset: categories}
    classification_models = [ModelTypes.ROBERTA]
    train_params = {ModelTypes.ROBERTA: {"metric": "f1"}}
    active_learning_strategies = [ActiveLearningStrategies.DAL, ActiveLearningStrategies.RANDOM, ActiveLearningStrategies.HARD_MINING, 
                                  ActiveLearningStrategies.DROPOUT_PERCEPTRON]

    experiments_runner = ExperimentRunnerImbalanced(first_model_positives_num=5,
                                                    first_model_negatives_num=5,
                                                    active_learning_suggestions_num=args.active_learning_suggestions_num)

    results_file_path, results_file_path_aggregated = res_handler.get_results_files_paths(
        experiment_name=experiment_name, start_timestamp=start_timestamp, repeats_num=num_experiment_repeats)

    for dataset in datasets_and_categories:
        for category in datasets_and_categories[dataset]:
            category_label = category.replace(' ', '').replace('/', '')
            for model in classification_models:
                results_all_repeats = defaultdict(lambda: defaultdict(list))
                for repeat in range(1, num_experiment_repeats + 1):
                    config = ExperimentParams(
                        experiment_name=experiment_name,
                        train_dataset_name=dataset + '_train',
                        dev_dataset_name=dataset + '_dev',
                        test_dataset_name=dataset + '_test',
                        category_name=category,
                        workspace_id=f'{experiment_name}-{dataset}-{category_label}-{model.name}-{repeat}',
                        model=model,
                        active_learning_strategies=active_learning_strategies,
                        repeat_id=repeat,
                        train_params=train_params[model],
                        cluster_path=args.cluster_path,
                        start_with_medoids=args.start_with_medoids,
                        ckpt_path = args.ckpt_path
                    )

                    # key: active learning name, value: dict with key: iteration number, value: results dict
                    results_per_active_learning = \
                        experiments_runner.run(config,
                                               active_learning_iterations_num=active_learning_iterations_num,
                                               results_file_path=results_file_path,
                                               delete_workspaces=True)
                    for al in results_per_active_learning:
                        for iteration in results_per_active_learning[al]:
                            results_all_repeats[al][iteration].append(results_per_active_learning[al][iteration])
                    for file in os.listdir("/home/mamooler/models"):
                        import shutil
                        file_path = os.path.join("/home/mamooler/models",file)
                        if os.path.isdir(file_path):
                             shutil.rmtree(file_path)
                        elif os.path.isfile(file_path):
                            os.remove(file_path)
                        
                   
                # aggregate the results of a single active learning iteration over num_experiment_repeats
                if num_experiment_repeats > 1:
                    agg_res_dicts = res_handler.avg_res_dicts(results_all_repeats)
                    res_handler.save_results(results_file_path_aggregated, agg_res_dicts)
