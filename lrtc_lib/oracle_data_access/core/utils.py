# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import ujson as json
import numpy as np
import os
import random
from typing import Mapping

from  definitions import ROOT_DIR
from  definitions import PROJECT_PROPERTIES
from  data_access.core.data_structs import nested_default_dict, Label

from clustering.cluster_statistic import cluster_statistic
import pandas as pd

gold_labels_per_dataset: (str, nested_default_dict()) = None  # (dataset, URIs -> categories -> Label)


def get_gold_labels_dump_dir():
    return os.path.join(ROOT_DIR, 'data', 'oracle_access_dumps')


def get_labels_dump_filename(dataset_name: str):
    return os.path.join(get_gold_labels_dump_dir(), dataset_name + '.json')


def get_gold_labels(dataset_name: str,  category_name: str = None) -> Mapping[str, Mapping[str, Label]]:
    """
    :param dataset_name: the name of the dataset from which the gold labels should be retrieved
    :param category_name: the name of the category for which label information is needed. Default is None, meaning all
    categories.
    :return: # URIs -> categories -> Label
    """
    global gold_labels_per_dataset

    if gold_labels_per_dataset is None or gold_labels_per_dataset[0] != dataset_name:  # not in memory
        if os.path.exists(get_labels_dump_filename(dataset_name)):  # try to load from disk
            with open(get_labels_dump_filename(dataset_name)) as json_file:
                text_and_gold_labels_encoded = json_file.read()
            simplified_dict = json.loads(text_and_gold_labels_encoded)
            labels_dict = {k: {category: Label(**label_dict) for category, label_dict in v.items()}
                           for k, v in simplified_dict.items()}
            gold_labels_per_dataset = (dataset_name, labels_dict)
        else:  # or create an empty in-memory
            gold_labels_per_dataset = (dataset_name, nested_default_dict())

    uri_categories_and_labels_map = gold_labels_per_dataset[1]
    if category_name is not None:
        data_view_func = PROJECT_PROPERTIES["data_view_func"]
        uri_categories_and_labels_map = data_view_func(category_name, uri_categories_and_labels_map)
    return uri_categories_and_labels_map


def sample(dataset_name: str, category_name: str, sample_size: int, random_seed: int, restrict_label: str = None, medoids_only=False, cluster_path:str=""):
    """
    return a sample of TextElements uris, for the given category in the given dataset, with their gold labels
    information. If restrict_label is provided - only TextElements with that label will be included.

    :param dataset_name: the name of the dataset from which TextElements should be retrieved
    :param category_name: the name of the category whose label information is the target of this sample
    :param sample_size: how many TextElements should be sampled
    :param random_seed: a seed for the Random being used for sampling
    :param restrict_label: restrict returning TextElements to elements with the given label.
    Default is None - do not avoid any label, i.e. sample from all TextElements.
    :return: a list of tuples of TextElement uri and a dictionary of categories to Labels.
    """

    def stat_round(samples):     
        pos = 0
        neg = 0
        i = 0
        nb_act = 0
        while pos<5 or neg<5:
            ind = random.randint(0,len(samples)-1)
            new_sample = samples[ind]#random.sample(tmp_new_all_gold_label_tuples, k=1)[0]
            samples.pop(ind)
            nb_act+=1
            i+=1
            if restrict_label in new_sample[1][category_name].labels:
                pos+=1
            else:
                neg+=1
        return nb_act

    def get_stats(candidate_list):
        random.shuffle(candidate_list)
        nb_actions = []
        for r in range(1000):
            candidate_samples = candidate_list.copy()
            nb_actions.append(stat_round(candidate_samples))

        nb_actions = np.array(nb_actions)
        median = np.median(nb_actions)
        percentile = np.percentile(nb_actions, 90)
        logging.info(f"median = {median}, 90% percentile = {percentile}")

    if sample_size <= 0:
        raise ValueError(f'There is no logic in sampling {sample_size} elements')

    gold_labels = get_gold_labels(dataset_name, category_name)
    all_gold_label_tuples = [(uri, label_dict) for uri, label_dict in gold_labels.items()]

 

    import  data_access.data_access_factory as data_access_factory  
    data_access = data_access_factory.get_data_access()
    all_text_elements = data_access.get_all_text_elements(dataset_name)
    # restrict by label
    if restrict_label is not None:
        gold_label_tuples = [t for t in all_gold_label_tuples if restrict_label in t[1][category_name].labels]
        
        gold_label_uris = [uri for (uri, label) in gold_label_tuples]
        all_gold_label_uris = [uri for (uri, label) in all_gold_label_tuples]

        wanted_text_elements = [elem for elem in all_text_elements if elem.uri in gold_label_uris]
        all_wanted_text_elements = [elem for elem in all_text_elements if elem.uri in all_gold_label_uris]

    else:
        gold_label_tuples = all_gold_label_tuples


    if medoids_only:

        logging.info("Getting the initial samples from mediods only")
      
        data = pd.read_csv(cluster_path)
        medoids_indices = np.load(cluster_path[:-4].replace("clusters", "stats", 1)+"_medoid_indices.npy")
        # only consider the clusters that are dense enough
        medoid_texts = [data.loc[index, "text"] for index in medoids_indices]
        wanted_medoid_uris = [elem.uri for elem in wanted_text_elements if elem.text in medoid_texts]
        all_wanted_medoid_uris = [elem.uri for elem in all_wanted_text_elements if elem.text in medoid_texts]
       

        old_count = len(gold_label_tuples)
        new_all_gold_label_tuples = [(uri, label_dict) for (uri, label_dict) in all_gold_label_tuples if uri in all_wanted_medoid_uris]
        new_gold_label_tuples = [(uri, label_dict) for (uri, label_dict) in gold_label_tuples if uri in wanted_medoid_uris]
        
        pos_med = 0
        for elem in new_all_gold_label_tuples:
            if restrict_label in elem[1][category_name].labels:
                pos_med += 1
        print(pos_med)
        print(len(new_all_gold_label_tuples))
        logging.info(f"standard sampling:")
        get_stats(all_gold_label_tuples)
        logging.info("Medoid sampling:")
        get_stats(new_all_gold_label_tuples)
       
    
        
        if len(new_gold_label_tuples)>0:
            gold_label_tuples = new_gold_label_tuples
            new_count = len(gold_label_tuples)
            logging.info(f"shrunk the choices from {old_count} to {new_count}")
        else:
            logging.info(f"{len(new_gold_label_tuples)} < {sample_size} => not enough mediods with the target label. We use the entire dataset instead.")
        

    # sample
    random.Random(random_seed).shuffle(gold_label_tuples)
    return gold_label_tuples[:min(sample_size, len(gold_label_tuples))]

