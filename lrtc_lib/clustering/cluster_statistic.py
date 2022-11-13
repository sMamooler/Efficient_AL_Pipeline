
import sklearn
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples
# from jqm_cvi.jqmcvi import base
import json
import pickle


class cluster_statistic():

    def __init__(self, cluster_path, dunn_index_threshold=0.2):
 
        self.data = pd.read_csv(cluster_path)
        print("read csv file")
        self.dunn_index_threshold = dunn_index_threshold

        self.cluster_centers = np.load(cluster_path[:-4].replace("clusters", "stats", 1)+"_centers.npy")
        print("loaded centers")
        with open(cluster_path[:-4].replace("clusters", "stats", 1)+"_embeddings.npz", "rb") as file:
            print("loading embeddings")
            self.embeddings = np.load(file)
        print("loaded embeddings")
        # print(self.embeddings)
        # convert embeddings from string to float 
        # if (type(self.data['embedding'].tolist()[0])==str):
        #     embeddings = []
        #     for embed in self.data['embedding']:
        #         embed = embed[1:-1].split()
        #         new_embed = []
        #         for e in embed:
        #             new_embed.append(float(e))

        #         embeddings.append(new_embed)
        #     self.data['embedding'] = embeddings
        points = self.embeddings
        
        # points = np.zeros((len(embeddings),  len(embeddings[0])))
        # for i,vect in enumerate(embeddings):
        #     points[i] = vect


        labels = [c for c in self.data["cluster"].values]
        # self.dunn_indices = self.compute_dunn_index(points, labels)
        # with open(cluster_path[:-4].replace("clusters", "stats", 1)+"_dunn.npz", 'wb') as file:
        #     np.save(file, self.dunn_indices)
        # self.silhouette_scores = silhouette_samples(points, labels)
        
        
        # self.data_gb = self.data.groupby(by=["cluster"]).groups
        # self.stat = {}
        # self.cluster_size = []
        # self.get_cluster_homogenity()
        # self.num_clusters = len(self.stat.columns)
        

        # # self.cluster_medoids = []
        # # self.get_medoids()
        # self.max_homogenities = np.zeros(self.num_clusters)
        # self.get_max_homogenity()
        # self.global_stat = {"homogenity":[], "num_clusters":[]}
        # self.get_global_stat()
        # self.labels = [int(l) for l in self.data["cluster"].values]

        
        

    def compute_dunn_index(self, points, labels):
        distances = euclidean_distances(points)
        ks = np.sort(np.unique(labels))
        
        deltas = np.ones([len(ks), len(ks)])*1000000
        big_deltas = np.zeros([len(ks), 1])
        
        l_range = list(range(0, len(ks)))
        
        
        for k in l_range:
            for l in (l_range[0:k]+l_range[k+1:]):
                deltas[k, l] = base.delta_fast((labels == ks[k]), (labels == ks[l]), distances)
            
            big_deltas[k] = base.big_delta_fast((labels == ks[k]), distances)
            

        di = np.min(deltas, axis=1)/np.max(big_deltas)
        
        return di

    def get_medoids(self):
        for cluster_id  in range(len(self.data_gb)):
            cluster = self.data_gb[cluster_id]
            embeddings = self.embeddings[int(cluster_id)].tolist()
            dists = euclidean_distances(embeddings, embeddings)
            medoid_indx = np.argmin(dists.sum(axis=0))
            #medoid = embeddings[medoid_indx]
            self.cluster_medoids.append(cluster[medoid_indx])


    def get_cluster_homogenity(self):
        for cluster_id in range(len(self.data_gb)):
            cluster = self.data_gb[cluster_id]
            cluster_stat = defaultdict(lambda:0)
            for index in cluster:
                cluster_stat[self.data.loc[index , :]["label"]]+=1/len(cluster)
            self.cluster_size.append(len(cluster))
            self.stat[cluster_id] = cluster_stat

        self.stat = pd.DataFrame(self.stat)
        self.stat = self.stat.fillna(0)
        return self.stat
        

    def get_max_homogenity(self):
        
        for i,c in enumerate(self.stat):
            self.max_homogenities[i] = self.stat[c].max()
        return self.max_homogenities
           
    def get_global_stat(self, thresholds=[0.8, 0.7, 0.6, 0.5]):
        for t in thresholds:
            self.global_stat["homogenity"].append(t)
            target_clusters = np.argwhere(self.max_homogenities>=t).flatten()
            
            
            sentences =  pd.Series(dtype=object)
            for c in target_clusters:
                sent_id = self.data_gb[c]
                sentences = pd.concat([sentences, self.data.loc[sent_id, :]["text"]])
            

            self.global_stat["num_clusters"].append(len(target_clusters))
            
        self.global_stat = pd.DataFrame(self.global_stat, columns=["homogenity", "num_clusters"])
        return self.global_stat



    def get_avg_homogenity(self):
        return self.max_homogenities.mean()



        
        

        
        

