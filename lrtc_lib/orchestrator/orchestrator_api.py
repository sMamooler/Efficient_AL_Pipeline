# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from email.policy import default
import glob
from importlib.util import spec_from_file_location
import logging
from operator import index
import os
import traceback
from collections import Counter, defaultdict
from enum import Enum
from typing import Mapping, List, Sequence, Tuple, Set
from numpy import double
import pandas as pd
import random

from sklearn import neighbors
import tensorflow as tf

import  data_access.data_access_factory as data_access_factory
from  active_learning.strategies import ActiveLearningStrategy
from  data_access.core.data_structs import Label, TextElement
from  data_access.core.utils import get_workspace_labels_dump_filename
from  definitions import PROJECT_PROPERTIES
from  orchestrator.core.state_api import orchestrator_state_api
from  orchestrator.core.state_api.orchestrator_state_api import ModelInfo, ActiveLearningRecommendationsStatus
from  train_and_infer_service.model_type import ModelType
from  train_and_infer_service.train_and_infer_api import ModelStatus
from  training_set_selector import training_set_selector_factory
from active_learning.strategies import ActiveLearningStrategies
import  oracle_data_access.core.utils as oracle_utils

from clustering.cluster_statistic import cluster_statistic
# constants

MAX_VALUE = 1000000

TRAIN_COUNTS_STR_KEY = "train_counts"
DEV_COUNTS_STR_KEY = "dev_counts"

LABEL_POSITIVE = "true"
LABEL_NEGATIVE = "false"
BINARY_LABELS = frozenset({LABEL_NEGATIVE, LABEL_POSITIVE})

# members

active_learning_strategy = PROJECT_PROPERTIES["active_learning_strategy"]
training_set_selection_strategy = PROJECT_PROPERTIES["training_set_selection"]
active_learner = PROJECT_PROPERTIES["active_learning_factory"].get_active_learner(active_learning_strategy)

data_access = data_access_factory.get_data_access()

train_and_dev_sets_selector = training_set_selector_factory.get_training_set_selector(
    selector=training_set_selection_strategy)


def _delete_orphan_labels():
    """
    delete labels that are not attached to a known workspace
    """
    all_label_dump_files = glob.glob(get_workspace_labels_dump_filename(workspace_id='*', dataset_name='*'))
    existing_workspace_ids = [w.workspace_id for w in orchestrator_state_api.get_all_workspaces()]
    dump_files_with_parents = [file for wid in existing_workspace_ids for file in
                               glob.glob(get_workspace_labels_dump_filename(workspace_id=wid, dataset_name='*'))]
    for dump_file in all_label_dump_files:
        if dump_file not in dump_files_with_parents:
            logging.info(f"deleting orphan labels file {dump_file}")
            os.remove(dump_file)


_delete_orphan_labels()


def copy_workspace(existing_workspace_id: str, new_workspace_id: str):
    """
    Creates a copy of a given workspace with its labels under a new workspace id
    :param existing_workspace_id:
    :param new_workspace_id:
    :return:
    """
    workspace = get_workspace(existing_workspace_id)
    dataset_name = workspace.dataset_name
    dev_dataset_name = workspace.dev_dataset_name
    data_access.copy_labels_to_new_workspace(existing_workspace_id, new_workspace_id, dataset_name, dev_dataset_name)
    orchestrator_state_api.copy_workspace(existing_workspace_id, new_workspace_id)
    return new_workspace_id


def set_training_set_selection_strategy(new_training_set_selection_strategy=None):
    """
    Set the logic for selecting training examples from the training dataset.
    The default strategy is ALL_LABELED, which means we use all the labeled elements
    other strategies enable to add weak labels, for example by using unlabeled elements as weak negative
    :return:
    """
    global train_and_dev_sets_selector
    global training_set_selection_strategy
    if new_training_set_selection_strategy is not None:
        training_set_selection_strategy = new_training_set_selection_strategy
        train_and_dev_sets_selector = training_set_selector_factory.get_training_set_selector(
            selector=training_set_selection_strategy)


def set_active_learning_strategy(new_active_learning_strategy=None):
    """
    Set active learning policy to use
    :param new_active_learning_strategy:
    :param second_active_learning_strategy (only used for cluster and match AL strategy):
    :return:
    """
    global active_learner, active_learning_strategy
    if new_active_learning_strategy is not None:
        active_learning_strategy = new_active_learning_strategy
        active_learner = PROJECT_PROPERTIES["active_learning_factory"].get_active_learner(active_learning_strategy)
        


def create_workspace(workspace_id: str, dataset_name: str, dev_dataset_name: str = None, test_dataset_name: str = None):
    """
    create a new workspace
    :param workspace_id:
    :param dataset_name:
    :param dev_dataset_name:
    :param test_dataset_name:
    """
    orchestrator_state_api.create_workspace(workspace_id, dataset_name, dev_dataset_name, test_dataset_name)
    logging.info(f"Creating workspace {workspace_id} using dataset {dataset_name}")


def create_new_category(workspace_id: str, category_name: str, category_description: str,
                        category_labels: Set[str] = BINARY_LABELS):
    """
    declare a new category in the given workspace
    :param workspace_id:
    :param category_name:
    :param category_description:
    :param category_labels:
    """
    orchestrator_state_api.add_category_to_workspace(workspace_id, category_name, category_description, category_labels)


class DeleteModels(Enum):
    ALL = 0
    FALSE = 1
    ALL_BUT_FIRST_MODEL = 2


def delete_workspace(workspace_id: str, checkpoint_path:str, delete_models: DeleteModels = DeleteModels.ALL, ignore_errors=False):
    """
    delete a given workspace
    :param workspace_id:
    :param delete_models: ALL - delete all the models of the workspace, FALSE - do not delete models,
    ALL_BUT_FIRST_MODEL - keep the first model of each category
    :param ignore_errors:
    """
    logging.info(f"deleting workspace {workspace_id} ignore errors {ignore_errors}")
    models_to_delete = []
    if workspace_exists(workspace_id):
        try:
            workspace = orchestrator_state_api.get_workspace(workspace_id)
            if delete_models != DeleteModels.FALSE:
                for category in workspace.category_to_models:
                    for idx, model_id in enumerate(workspace.category_to_models[category]):
                        if idx == 0 and delete_models == DeleteModels.ALL_BUT_FIRST_MODEL:
                            continue
                        models_to_delete.append(_get_model(workspace_id, model_id))
            orchestrator_state_api.delete_workspace_state(workspace_id)
        except Exception as e:
            logging.error(f"error deleting workspace {workspace_id}")
            traceback.print_exc()
            if not ignore_errors:
                raise e
        try:
            data_access.clear_saved_labels(workspace_id, workspace.dataset_name)
            if workspace.dev_dataset_name:
                data_access.clear_saved_labels(workspace_id, workspace.dev_dataset_name)
        except Exception as e:
            logging.error(f"error clearing saved label for workspace {workspace_id}")
            traceback.print_exc()
            if not ignore_errors:
                raise e
    for model in models_to_delete:
        model_type = model.model_type
        train_and_infer = PROJECT_PROPERTIES["train_and_infer_factory"].get_train_and_infer(model_type, checkpoint_path)
        train_and_infer.delete_model(model.model_id)

def get_active_learner_per_element_score(items: Sequence[TextElement], workspace_id: str, checkpoint_path:str, dataset_name: str, category_name: str):
    """
    returns the scores of the AL strategy for the given items
    :param items: 
    :param workspace_id:
    :param model_id:
    :param dataset_name:
    :param category_name:
    :return:
    """
    model_id = orchestrator_state_api.get_latest_model_id_by_al_status(workspace_id, category_name, ModelStatus.READY)
    all_scores = active_learner.get_per_element_score(items, workspace_id, model_id, checkpoint_path, dataset_name, category_name)
    return all_scores

def compute_uncertainty_embeddings(workspace_id: str, dataset_name: str, category_name: str, model: ModelType, train_params):
    # import uuid
    # orchestrator_state_api.add_model(workspace_id=workspace_id, category_name=category_name, model_id=str(uuid.uuid1()),
    #                                  model_status=ModelStatus.READY, model_type=model, model_metadata=train_params)
    
    # unlabeled samples
    from data_access.data_access_factory import get_data_access
    data_access = get_data_access()
    max_to_consider = 10 ** 6
    unlabeled_text_elements = data_access.sample_unlabeled_text_elements(workspace_id, dataset_name, category_name,
                                                               max_to_consider, remove_duplicates=True)["results"]
    #unlabeled_text_elements = get_unlabele(dataset_name)
    all_scores = get_active_learner_per_element_score(unlabeled_text_elements, workspace_id, dataset_name, category_name)
  
    embeddings =  infer(workspace_id, category_name, unlabeled_text_elements)["embeddings"]
    uncertainty_embeddings = tf.math.multiply(tf.expand_dims(all_scores, axis=1), embeddings)
    normalized_uncertainty_embeddings = tf.math.l2_normalize(uncertainty_embeddings, axis=1).numpy()
    # clustering
    from sklearn.cluster import KMeans
    k = len(unlabeled_text_elements)//5
    model = KMeans(k)
    kmeans = model.fit(normalized_uncertainty_embeddings)
    clusters = kmeans.labels_
    
    # store clusters
    elem_2_cluster = {}
    cluster_2_elem = defaultdict(lambda: [])
    for i,e in enumerate(unlabeled_text_elements):
        elem_2_cluster[e.uri] = clusters[i]
        cluster_2_elem[clusters[i]].append(e.uri)
    

    return elem_2_cluster, cluster_2_elem



def edit_category(workspace_id: str, prev_category_name: str, new_category_name: str, new_category_description: str):
    raise Exception("Not implemented yet")


def delete_category(workspace_id: str, category_name: str):
    raise Exception("Not implemented yet")


def add_documents(dataset_name, docs):
    data_access.add_documents(dataset_name=dataset_name, documents=docs)


def query(workspace_id: str, dataset_name: str, category_name: str, query: str,
          sample_size: int,  unlabeled_only: bool = False, remove_duplicates=False) -> Mapping[str, object]:
    """
    query a dataset using the given regex, returning up to *sample_size* elements that meet the query

    :param workspace_id:
    :param dataset_name:
    :param category_name:
    :param query: regex string
    :param unlabeled_only: if True, filters out labeled elements
    :param sample_size: maximum items to return
    :param remove_duplicates: if True, remove duplicate elements
    :return: a dictionary with two keys: 'results' whose value is a list of TextElements, and 'hit_count' whose
    value is the total number of TextElements in the dataset matched by the query.
    {'results': [TextElement], 'hit_count': int}
    """

    if unlabeled_only:
        return data_access.sample_unlabeled_text_elements(workspace_id=workspace_id, dataset_name=dataset_name,
                                                          category_name=category_name, sample_size=sample_size,
                                                          query=query, remove_duplicates=remove_duplicates)
    else:
        return data_access.sample_text_elements_with_labels_info(workspace_id=workspace_id, dataset_name=dataset_name,
                                                                 sample_size=sample_size, query=query,
                                                                 remove_duplicates=remove_duplicates)


def get_documents(workspace_id: str, dataset_name: str, uris: Sequence[str]) -> List[object]:
    """
    :rtype: list of Document
    :param workspace_id:
    :param dataset_name:
    :param uris:
    """
    return data_access.get_documents_with_labels_info(workspace_id, dataset_name, uris)


def get_text_elements(workspace_id: str, dataset_name: str, uris: Sequence[str]) -> List[object]:
    """
    :param workspace_id:
    :param dataset_name:
    :param uris:
    """
    return data_access.get_text_elements_with_labels_info(workspace_id, dataset_name, uris)


def _update_recommendation(workspace_id, dataset_name, category_name, checkpoint_path, count, model: ModelInfo = None, view_dir="", medoids_only=False):
    """
    Using the AL strategy, update the workspace with next recommended elements for labeling
    :param workspace_id:
    :param dataset_name:
    :param category_name:
    :param count:
    :param model: model to use or None to use the latest model in status READY
    """
    if model is None:
        model = orchestrator_state_api.get_latest_model_by_state(workspace_id, category_name, ModelStatus.READY)
    curr_cat_recommendations = orchestrator_state_api.get_current_category_recommendations(workspace_id, category_name,
                                                                                           model.model_id)
    num_recommendations = len(curr_cat_recommendations)
    if num_recommendations < count:
        orchestrator_state_api.update_active_learning_status(workspace_id, category_name, model.model_id,
                                                             ActiveLearningRecommendationsStatus.AL_IN_PROGRESS)

        new_recommendations = active_learner.get_recommended_items_for_labeling(
            workspace_id=workspace_id, model_id=model.model_id, checkpoint_path=checkpoint_path, dataset_name=dataset_name, category_name=category_name, medoids_only=medoids_only,
            sample_size=count)
        orchestrator_state_api.update_category_recommendations(workspace_id=workspace_id, category_name=category_name,
                                                               model_id=model.model_id,
                                                               recommended_items=new_recommendations)
        orchestrator_state_api.update_active_learning_status(workspace_id, category_name, model.model_id,
                                                             ActiveLearningRecommendationsStatus.READY)
    return model.model_id


def get_model_active_learning_status(workspace_id, model_id):
    return orchestrator_state_api.get_active_learning_status(workspace_id, model_id)


def get_elements_to_label(workspace_id: str, category_name: str, checkpoint_path:str, count: int, view_dir="", medoids_only=False) -> Sequence[TextElement]:
    """
    returns a list of the top *count* elements recommended for labeling by the AL strategy.
    The active learner is invoked only if the requested count of elements have not yet been added to the workspace.
    :param workspace_id:
    :param category_name:
    :param count:
    """
    dataset_name = get_workspace(workspace_id).dataset_name
    model_id = _update_recommendation(workspace_id, dataset_name, category_name, checkpoint_path, count, medoids_only=medoids_only)
    updated_recommended = \
        orchestrator_state_api.get_current_category_recommendations(workspace_id, category_name, model_id)
    return updated_recommended


    

def set_labels(workspace_id: str, labeled_sentences: Sequence[Tuple[str, Mapping[str, Label]]],
               propagate_to_duplicates=False):
    """
    set labels for URIs.
    :param workspace_id:
    :param labeled_sentences: Sequence of tuples of URI and a dict in the format of {"category_name":Label},
    where Label is an instance of data_structs.Label
    :param propagate_to_duplicates: if True, also set the same labels for additional URIs that are duplicates of
    the URIs provided.
    """

    return_value = data_access.set_labels(workspace_id, labeled_sentences, propagate_to_duplicates)
    return return_value


def unset_labels(workspace_id: str, category_name, uris: Sequence[str]):
    """
    unset labels of a given category for URIs.
    :param workspace_id:
    :param category_name:
    :param uris:
    """
    data_access.unset_labels(workspace_id, category_name, uris)


def _convert_to_dicts_with_numeric_labels(data, category_name, all_category_labels: Set[str]) -> Sequence[Mapping]:
    """
    convert textual labels to integers and convert to expected inference input format
    :param data:
    """
    text_to_number = {label: i for i, label in enumerate(sorted(all_category_labels))}

    def get_numeric_value(labels_set):
        if len(labels_set) == 1:
            return text_to_number[next(iter(labels_set))]
        else:
            raise ValueError("multilabel is not supported currently")

    converted_data = [{"text": element.text,
                       "label": get_numeric_value(element.category_to_label[category_name].labels)}
                      for element in data]
    return converted_data


def first_iteration_pseudo_label(workspace_id: str, category_name: str, suggested_text_elements, cosine_similarity_threshold, view_dir):
    """
    train a model for a category in the specified workspace
    :param workspace_id:
    :param category_name:
    :param suggested_text_elements: the text elements (1 positive, 0 negative) provided by the user at the first iteration
    :param view_dir: path of the csv file containing the clsuter views
    :return pseudo_labeld_samples: list of pseudo-labeled TextElements
    """
    workspace = get_workspace(workspace_id)
    dataset_name = workspace.dataset_name

    # get all text elements:
    data_access = data_access_factory.get_data_access()
    all_text_elements = data_access.get_all_text_elements(dataset_name)
    gold_labels = oracle_utils.get_gold_labels(dataset_name, category_name)
    pos = 0
    for uri, label in gold_labels.items():
        if label[category_name].labels==frozenset([LABEL_POSITIVE]):
            pos += 1

    logging.info(f"all positive counts: {pos}")

    view_names = []
    views = [] 
    view_file = "sbert-distilled-task-adapt_437-clusters.csv"
    #for view_file in os.listdir(view_dir+"clusters/"):
    name = view_file.split("_")[0]
    view_names.append(name)
    view = cluster_statistic(pd.read_csv(view_dir+"clusters/"+view_file), name, view_dir+"stats/")
    views.append(view)

    num_actions = 0
    # TODO: how to use multiple views when working with 5 samples?
    v = views[0]
    pseudo_labeled_samples = []
    # for each newly added phrase, get the cluster
    for text_element in suggested_text_elements:
        num_actions+=1
        ### PHASE 1: the recommended sample cluster ###
        current_cluster_id = v.data[v.data["text"]==text_element.text]["cluster"].values[0]
        
        indices = v.data_gb[current_cluster_id].to_list()
        homogeneity = v.stat.transpose()[category_name][current_cluster_id]

        cluster_medoid_index = v.cluster_medoids[current_cluster_id]
        cluster_elements = [all_text_elements[i] for i in indices]
        cluster_uris = [element.uri for element in cluster_elements]
        cluster_uris_with_labels = [(uri, gold_labels[uri]) for uri in cluster_uris if gold_labels[uri]]  
        cluster_medoid_text_element = all_text_elements[cluster_medoid_index]
        cluster_medoid_label= [label for uri, label in cluster_uris_with_labels if uri==cluster_medoid_text_element.uri][0]
        
        logging.info(f"{len(indices)} samples before prunning")
        indices = [index for index in indices if v.silhouette_scores[index]>0]
        logging.info(f"{len(indices)} samples after prunning")
        cluster_elements = [all_text_elements[i] for i in indices]
        cluster_uris = [element.uri for element in cluster_elements]
        cluster_uris_with_labels = [(uri, gold_labels[uri]) for uri in cluster_uris if gold_labels[uri]]  

        if cluster_medoid_label[category_name].labels==frozenset([LABEL_POSITIVE]):
            cluster_label = LABEL_POSITIVE
        else:
            cluster_label = LABEL_NEGATIVE
        
        
        pos = 0
        for uri, label in cluster_uris_with_labels:
            if label[category_name].labels==frozenset([LABEL_POSITIVE]):
                pos += 1

        logging.info(f"all positive counts in this cluster: {pos}")
        # get 5 random elements of the cluster and label them
        # five_random = random.choices(cluster_uris_with_labels, k=5)

        # # label the entire cluster based on majority voting
        # pos_count = 0
        # for (uri, label) in five_random:
        #     num_actions+=1
        #     if label[category_name].labels==frozenset([LABEL_POSITIVE]):
        #         pos_count += 1
        
        # set_labels(workspace_id, five_random)
        # logging.info(f"found {pos_count} positive samples in this cluster")
        # if pos_count>2:
        #     cluster_label = LABEL_POSITIVE
        # else:
        #     cluster_label = LABEL_NEGATIVE
        #     # return the empty list to ask for a new positive sample (this does not add to number of actions as we assume the user provides a positive sample in the first iteration)
        #     # NOTE: the caller of this funtion should check and repeat if the cosine_similarity threshold is 0
        #     break

        if homogeneity<0.8:
            continue
        else:
            for (uri, labels) in cluster_uris_with_labels: 
                #if (uri,labels) not in five_random:
                labels[category_name].labels = frozenset([cluster_label])
                pseudo_labeled_samples.append((uri, labels))
                
        
            ### PHASE 2: the neighboring clusters ###
            # get neighbors based on cosine similarity
            current_cluster_center = v.cluster_centers[current_cluster_id]
            m = tf.keras.metrics.CosineSimilarity(axis=0)
            cosine_similarities = []
            for cent in v.cluster_centers:
                m.update_state(current_cluster_center, cent)
                cosine_similarities.append(m.result().numpy())

            neighbors = tf.argsort(cosine_similarities, direction='DESCENDING')[1:].numpy()
            
            # check their medoid until you find a negative one
            for neighbor_id in neighbors:
                
                neighbor_cluster_indices = v.data_gb[neighbor_id].to_list()
                neighbor_medoid_index = v.cluster_medoids[neighbor_id]
                neighbor_medoid_text_element = all_text_elements[neighbor_medoid_index]
                neighbor_medoid_uri = neighbor_medoid_text_element.uri
                neighbor_medoid_label = gold_labels[neighbor_medoid_uri][category_name].labels

                if neighbor_medoid_label==frozenset([LABEL_POSITIVE]):

                    set_labels(workspace_id, [(neighbor_medoid_uri, gold_labels[neighbor_medoid_uri])])
                    num_actions+=1

                    neighbor_elements = [all_text_elements[i] for i in neighbor_cluster_indices]
                    neighbor_uris = [element.uri for element in neighbor_elements]
                    neighbor_uris_with_labels = [(uri, gold_labels[uri]) for uri in neighbor_uris if gold_labels[uri]]

                    for (uri, labels) in neighbor_uris_with_labels: 
                        if uri!=neighbor_medoid_uri:
                            labels[category_name].labels = frozenset([LABEL_POSITIVE])
                            pseudo_labeled_samples.append((uri, labels))
                            

                else:
                    # set the similarity threshold (by setting it in the configuration)
                    cosine_similarity_threshold = cosine_similarities[neighbor_id]
                    break

    return pseudo_labeled_samples, cosine_similarity_threshold, num_actions

def pseudo_label(workspace_id: str, category_name: str, suggested_text_elements, cosine_similarity_threshold, view_dir):
    """
    train a model for a category in the specified workspace
    :param workspace_id:
    :param category_name:
    :param suggested_text_elements: the text elements suggested by the al method in this iteration
    :param cosine_similarity_threshold: used to find the neighboring clusters
    :param cosine_similarity_threshold: required for speudo labeling
    :param view_dir: path of the csv file containing the clsuter views
    :return pseudo_labeld_samples: list of pseudo-labeled TextElements
    """
    workspace = get_workspace(workspace_id)
    dataset_name = workspace.dataset_name

    # get all text elements:
    data_access = data_access_factory.get_data_access()
    all_text_elements = data_access.get_all_text_elements(dataset_name)
    gold_labels = oracle_utils.get_gold_labels(dataset_name, category_name) 

    view_names = []
    views = []
    view_file = "sbert-distilled-task-adapt_437-clusters.csv"
    #for view_file in os.listdir(view_dir+"clusters/"):
    name = view_file.split("_")[0]
    view_names.append(name)
    view = cluster_statistic(pd.read_csv(view_dir+"clusters/"+view_file), name, view_dir+"stats/")
    views.append(view)

    num_actions = 0
    # TODO: how to use multiple views when working with 5 samples?
    v = views[0]
    pseudo_labeled_samples = []
    # for each newly added phrase, get the cluster
    for text_element in suggested_text_elements:
        num_actions += 1
        current_cluster_id = v.data[v.data["text"]==text_element.text]["cluster"].values[0]
        indices = v.data_gb[current_cluster_id].to_list()
        homogeneity = v.stat.transpose()[category_name][current_cluster_id]
        

        cluster_medoid_index = v.cluster_medoids[current_cluster_id]
        cluster_elements = [all_text_elements[i] for i in indices]
        cluster_uris = [element.uri for element in cluster_elements]
        cluster_uris_with_labels = [(uri, gold_labels[uri]) for uri in cluster_uris if gold_labels[uri]]  
        cluster_medoid_text_element = all_text_elements[cluster_medoid_index]
        cluster_medoid_label= [label for uri, label in cluster_uris_with_labels if uri==cluster_medoid_text_element.uri][0]
        
        logging.info(f"{len(indices)} samples before prunning")
        indices = [index for index in indices if v.silhouette_scores[index]>0]
        logging.info(f"{len(indices)} samples after prunning")
        cluster_elements = [all_text_elements[i] for i in indices]
        cluster_uris = [element.uri for element in cluster_elements]
        cluster_uris_with_labels = [(uri, gold_labels[uri]) for uri in cluster_uris if gold_labels[uri]]
        
        if cluster_medoid_label[category_name].labels==frozenset([LABEL_POSITIVE]):
            cluster_label = LABEL_POSITIVE
        else:
            cluster_label = LABEL_NEGATIVE

        # get 5 random elements of the cluster and label them
        # five_random = random.choices(cluster_uris_with_labels, k=5)

        # # label the entire cluster based on majority voting
        # pos_count = 0
        # for (uri, label) in five_random:
        #     num_actions += 1
        #     if label[category_name].labels==frozenset([LABEL_POSITIVE]):
        #         pos_count += 1

        # set_labels(workspace_id, five_random)

        # if pos_count>2:
        #     cluster_label = LABEL_POSITIVE
        # else:
        #     cluster_label = LABEL_NEGATIVE
        
        if homogeneity>=0.8:
            for (uri, labels) in cluster_uris_with_labels: 
                #if (uri,label) not in five_random:
                labels[category_name].labels = frozenset([cluster_label])
                pseudo_labeled_samples.append((uri, labels))
            

            ### PHASE 2: the neighboring clusters ###
            # get neighbors based on cosine similarity
            # current_cluster_center = v.cluster_centers[current_cluster_id]
            # m = tf.keras.metrics.CosineSimilarity(axis=0)
            # cosine_similarities = []
            # for cent in v.cluster_centers:
            #     m.update_state(current_cluster_center, cent)
            #     cosine_similarities.append(m.result().numpy())

            # neighbors = tf.argsort(cosine_similarities, direction='DESCENDING')[1:].numpy()
            
            # # speudo label neighbors until the cosine similarity is smaller than the threshold
            # for neighbor_id in neighbors:
                
            #     if cosine_similarities[neighbor_id]>=cosine_similarity_threshold:
            #         neighbor_cluster_indices = v.data_gb[neighbor_id].to_list()
            #         neighbor_elements = [all_text_elements[i] for i in neighbor_cluster_indices]
            #         neighbor_uris = [element.uri for element in neighbor_elements]
            #         neighbor_uris_with_labels = [(uri, gold_labels[uri]) for uri in neighbor_uris if gold_labels[uri]]

            #         for (uri, labels) in neighbor_uris_with_labels: 
            #             labels[category_name].labels = frozenset([cluster_label])
            #             pseudo_labeled_samples.append((uri, labels))

            #     else:
            #         break

    return pseudo_labeled_samples, cosine_similarity_threshold, num_actions
    


def remove_redundant_samples(workspace_id: str, suggested_text_elements, view_dir):
    """
    remove samples similar to the recommended one from the data as they won't contribute
    :param workspace_id:
    :param category_name:
    :param suggested_text_elements: the text elements suggested by the al method in this iteration
    :param cosine_similarity_threshold: used to find the neighboring clusters
    :param cosine_similarity_threshold: required for speudo labeling
    :param view_dir: path of the csv file containing the clsuter views
    :return pseudo_labeld_samples: list of pseudo-labeled TextElements
    """
    workspace = get_workspace(workspace_id)
    dataset_name = workspace.dataset_name

    # get all text elements:
    data_access = data_access_factory.get_data_access()
    all_text_elements = data_access.get_all_text_elements(dataset_name)

    view_names = []
    views = []
    view_file ="sbert-distilled-task-adapt_437-clusters.csv"
    #for view_file in os.listdir(view_dir+"clusters/"):
    name = view_file.split("_")[0]
    view_names.append(name)
    view = cluster_statistic(pd.read_csv(view_dir+"clusters/"+view_file), name, view_dir+"stats/")
    views.append(view)

    num_actions = 0
   
    v = views[0]

    # for each newly added phrase, get the cluster
    for text_element in suggested_text_elements:
        num_actions += 1
        current_cluster_id = v.data[v.data["text"]==text_element.text]["cluster"].values[0]
        indices = v.data_gb[current_cluster_id].to_list()
        # find samples with high Silhouette index
        high_silhouette_scores_indices = [i for i in indices if v.silhouette_scores[i]>0.1]
        cluster_elements = [all_text_elements[i] for i in high_silhouette_scores_indices]
        cluster_uris = [element.uri for element in cluster_elements]

        if len(cluster_uris)>0:
            # remove them from data as they provide no additional information 
            logging.info(f"removing {len(cluster_uris)} redundant samples")

            logging.info(f"{len(data_access.get_all_text_elements(dataset_name))} samples before removing redundant samples")
            data_access.remove_from_data(workspace_id, cluster_uris)
            logging.info(f"{len(data_access.get_all_text_elements(dataset_name))} samples after removing redundant samples")
        
      

def train(workspace_id: str, category_name: str, model_type: ModelType, checkpoint_path: str, train_params=None, infer_after_train=True, first_iteration=False,
            get_pseudo_labels=False, medoids_only=False, view_dir="", suggested_text_elements=[], cosine_similarity_threshold=0
        ):
    """
    train a model for a category in the specified workspace
    :param workspace_id:
    :param category_name:
    :param model_type:
    :param suggested_text_elements: the text elements suggested by the al method in this iteration
    :param train_params:
    :param infer_after_train:
    :param pseudo_label:
    :param view_dir: path of the csv file containing the clsuter views
    :param pseudo_label: if True, pseudo-labelled samples will be added using cluster and match method
    :param cosine_similarity_threshold: required for speudo labeling
    :return: model id
    :return: cosine_similarity_threshold 
    :return: num_actions:
    :return: pseudo_labeled_samples:
    """
    workspace = get_workspace(workspace_id)
    dataset_name = workspace.dataset_name

    pseudo_labeled_samples = []
    if get_pseudo_labels:
        #assert len(suggested_text_elements)>0 and view_dir!="", "suggested_text_elements and views are required for speudo labeling"
        if first_iteration:
            pseudo_labeled_samples, cosine_similarity_threshold, num_actions = first_iteration_pseudo_label(workspace_id, category_name, suggested_text_elements, cosine_similarity_threshold, view_dir)
            if cosine_similarity_threshold==0:
                logging.info("the provided sample for the first iteration is not positive, trying again ...")
                return 0, cosine_similarity_threshold, num_actions, pseudo_labeled_samples
        else:
            assert cosine_similarity_threshold>0, "cosine similarity should be provided for speudo labeling at iterations higher than 1"
            pseudo_labeled_samples, cosine_similarity_threshold, num_actions = pseudo_label(workspace_id, category_name, suggested_text_elements, cosine_similarity_threshold, view_dir)
        if len(pseudo_labeled_samples)>0:
            set_labels(workspace_id, pseudo_labeled_samples)
    else:
        num_actions = len(suggested_text_elements)
        
    # if medoids_only:
    #     remove_redundant_samples(workspace_id, suggested_text_elements, view_dir)

    (train_data, train_counts), (dev_data, dev_counts) = train_and_dev_sets_selector.get_train_and_dev_sets(
        workspace_id=workspace_id, train_dataset_name=dataset_name, category_name=category_name,
        dev_dataset_name=workspace.dev_dataset_name)
    logging.info(f"training a new model with {train_counts}")
 

    # label_counts != train_counts as train_counts may refer to negative and weak negative labels separately
    labels = [element.category_to_label[category_name].labels for element in train_data]
    labels = [item for subset in labels for item in subset]  # flatten list of sets
    label_counts = Counter(labels)
    all_category_labels = workspace.category_to_labels[category_name]
    labels_not_in_train = [label for label in all_category_labels if label_counts[label] == 0]
    if len(labels_not_in_train) > 0:
        raise Exception(f"no train examples for labels: {labels_not_in_train}, cannot train a model: {train_counts}")

    model_metadata = dict()
    model_metadata[TRAIN_COUNTS_STR_KEY] = train_counts
    if dev_data is not None:
        model_metadata[DEV_COUNTS_STR_KEY] = dev_counts

    logging.info(
        f"workspace {workspace_id} training a model for category '{category_name}', model_metadata: {model_metadata}")

    train_data = _convert_to_dicts_with_numeric_labels(train_data, category_name, all_category_labels)
    if dev_data:
        dev_data = _convert_to_dicts_with_numeric_labels(dev_data, category_name, all_category_labels)

    elements_to_infer = None
    if infer_after_train:  # add data to be inferred and cached after the training process
        test_dataset = data_access.sample_text_elements(workspace.test_dataset_name, MAX_VALUE)['results'] \
            if workspace.test_dataset_name is not None else []
        all_train_dataset = data_access.sample_text_elements(workspace.dataset_name, MAX_VALUE)['results']
        elements_to_infer = [{"text": element.text} for element in test_dataset + all_train_dataset]

    params = model_metadata if train_params is None else {**train_params, **model_metadata}
    train_and_infer = PROJECT_PROPERTIES["train_and_infer_factory"].get_train_and_infer(model_type, checkpoint_path)
    model_id = train_and_infer.train(train_data=train_data, dev_data=dev_data, test_data=elements_to_infer,
                                     train_params=params)
    logging.info(f"new model id is {model_id}")

    model_status = train_and_infer.get_model_status(model_id)
    orchestrator_state_api.add_model(workspace_id=workspace_id, category_name=category_name, model_id=model_id,
                                     model_status=model_status, model_type=model_type, model_metadata=params)
    
    if get_pseudo_labels and len(pseudo_labeled_samples)>0:
        # unset the pseudo labels after the training
        uris_to_unset = list(set([uri for uri, label in pseudo_labeled_samples]))
        unset_labels(workspace_id, category_name, uris_to_unset)
        
    return model_id, cosine_similarity_threshold, num_actions, pseudo_labeled_samples


def get_model_status(workspace_id: str, model_id: str) -> ModelStatus:
    """
    ModelStatus can be TRAINING, READY or ERROR
    :param workspace_id:
    :param model_id:
    :return:
    """
    model = _get_model(workspace_id, model_id)
    return model.model_status


def get_model_train_counts(workspace_id: str, model_id: str) -> Mapping:
    """
    number of elements for each label that were used to train a given model
    :param workspace_id:
    :param model_id:
    :return:
    """
    model = _get_model(workspace_id, model_id)
    return model.model_metadata[TRAIN_COUNTS_STR_KEY]


def get_all_models_for_category(workspace_id, category_name: str):
    """
    :param workspace_id:
    :param category_name:
    :return: dict from model_id to ModelInfo
    """
    workspace = get_workspace(workspace_id)
    return workspace.category_to_models.get(category_name, {})


def infer(workspace_id: str, category_name: str, checkpoint_path:str, texts_to_infer: Sequence[TextElement], model_id: str = None,
          infer_params: dict = None, use_cache: bool = True) -> dict:
    """
    get the prediction for a list of TextElements
    :param workspace_id:
    :param category_name:
    :param texts_to_infer: list of TextElements
    :param model_id: model_id to use. If set to None, the latest model for the category will be used
    :param infer_params: dictionary for additional inference parameters. Default is None
    :param use_cache: utilize a cache that stores inference results
    :return: a dictionary of inference results, with at least the "labels" key, where the value is a list of string
    labels for each element in texts_to_infer. Additional keys, with list values of the same length, can be passed.
    e.g. {"labels": ['false', 'true', 'true'],
          "scores": [0.23, 0.79, 0.98],
          "gradients": [[0.24, -0.39, -0.66, 0.25], [0.14, 0.29, -0.26, 0.16], [-0.46, 0.61, -0.02, 0.23]]}
    """
    models = get_all_models_for_category(workspace_id, category_name)
    if len(models) == 0:
        raise Exception(f"There are no models in workspace {workspace_id} for category {category_name}")
    if model_id is None:  # use latest
        model = orchestrator_state_api.get_latest_model_by_state(workspace_id=workspace_id,
                                                                 category_name=category_name,
                                                                 model_status=ModelStatus.READY)
    else:
        model = _get_model(workspace_id, model_id)
        if model.model_status is not ModelStatus.READY:
            raise Exception(f"model id {model_id} is not in READY status")

    train_and_infer = PROJECT_PROPERTIES["train_and_infer_factory"].get_train_and_infer(model.model_type, checkpoint_path)
    list_of_dicts = [{"text": element.text} for element in texts_to_infer]
    infer_results = train_and_infer.infer(model_id=model.model_id, items_to_infer=list_of_dicts,
                                          infer_params=infer_params, use_cache=use_cache)

    all_labels = get_workspace(workspace_id).category_to_labels[category_name]
    numeric_label_to_text = {i: label for i, label in enumerate(sorted(all_labels))}
    infer_results['labels'] = [numeric_label_to_text[l] for l in infer_results['labels']]
    return infer_results


def infer_by_uris(workspace_id: str, category_name: str, checkpoint_path:str, uris_to_infer: Sequence[str], model_id: str = None,
                  infer_params: dict = None, use_cache: bool = True) -> dict:
    """
    get the prediction for a list of URIs
    :param workspace_id:
    :param category_name:
    :param uris_to_infer: list of uris (str)
    :param model_id: model_id to use. If set to None, the latest model for the category will be used
    :param infer_params: dictionary for additional inference parameters. Default is None
    :param use_cache: utilize a cache that stores inference results
    :return: a dictionary of inference results, with at least the "labels" key, where the value is a list of string
    labels for each element in texts_to_infer. Additional keys, with list values of the same length, can be passed.
    e.g. {"labels": ['false', 'true', 'true'],
          "scores": [0.23, 0.79, 0.98],
          "gradients": [[0.24, -0.39, -0.66, 0.25], [0.14, 0.29, -0.26, 0.16], [-0.46, 0.61, -0.02, 0.23]]}
    """
    dataset_name = get_workspace(workspace_id).dataset_name
    elements_to_infer = data_access.get_text_elements_with_labels_info(workspace_id, dataset_name, uris_to_infer)
    return infer(workspace_id, category_name, checkpoint_path, elements_to_infer, model_id, infer_params, use_cache)


def get_all_text_elements(dataset_name: str) -> List[TextElement]:
    """
    get all the text elements of the given dataset
    :param dataset_name:
    """
    return data_access.get_all_text_elements(dataset_name=dataset_name)


def get_all_text_elements_uris(dataset_name: str) -> List[str]:
    """
    Return a List of all TextElement uris in the given dataset_name.
    :param dataset_name: the name of the dataset from which the TextElement uris should be retrieved.
    :return: a List of all TextElement uris in the given dataset_name.
    """
    return data_access.get_all_text_elements_uris(dataset_name=dataset_name)


def get_all_document_uris(workspace_id):
    dataset_name = get_workspace(workspace_id).dataset_name
    return data_access.get_all_document_uris(dataset_name)


def get_label_counts(workspace_id: str, dataset_name: str, category_name: str, remove_duplicates=True):
    """
    get the number of elements that were labeled.
    :param workspace_id:
    :param dataset_name:
    :param category_name:
    :param remove_duplicates: whether to count all labeled elements or only unique instances
    :return:
    """
    return data_access.get_label_counts(workspace_id, dataset_name, category_name, remove_duplicates=remove_duplicates)


def is_model_compatible_with_active_learning(al: ActiveLearningStrategy, model: ModelType):
    """
    return true if active learning strategy is supported by the given model type
    for example, ActiveLearningStrategies.CORE_SET and ActiveLearningStrategies.DAL are not supported by Naive Bayes
    defined in method get_compatible_models() under  active_learning.strategies.py
    :param al:
    :param model:
    :return:
    """
    return PROJECT_PROPERTIES["models_compatible_with_strategies_func"](model, al)


def delete_model_from_workspace(workspace_id, category_name, model_id, checkpoint_path):
    model_type = _get_model(workspace_id, model_id).model_type
    train_and_infer = PROJECT_PROPERTIES["train_and_infer_factory"].get_train_and_infer(model_type, checkpoint_path)
    logging.info(f"deleting model id {model_id} from workspace {workspace_id} in category {category_name}")
    orchestrator_state_api.delete_model(workspace_id, category_name, model_id)
    train_and_infer.delete_model(model_id)


def add_train_param(workspace_id: str, train_param_key: str, train_param_value: str):
    raise Exception("Not implemented yet")


def workspace_exists(workspace_id: str) -> bool:
    return orchestrator_state_api.workspace_exists(workspace_id)


def get_workspace(workspace_id):
    if not workspace_exists(workspace_id):
        raise Exception(f"workspace_id '{workspace_id}' doesn't exist")
    return orchestrator_state_api.get_workspace(workspace_id)


def _get_model(workspace_id, model_id):
    workspace = get_workspace(workspace_id)
    all_models = {k: v for d in workspace.category_to_models.values() for k, v in d.items()}
    if all_models[model_id]:
        return all_models[model_id]
    raise Exception(f"model id {model_id} does not exist in workspace {workspace_id}")
