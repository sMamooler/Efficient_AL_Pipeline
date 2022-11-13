import numpy as np
import torch
import pandas as pd
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from sentence_transformers import SentenceTransformer

def kmeans_cluster(text_encoder, dataframe, k, output_dir, method="pre-trained_sbert"):

    all_texts = np.array(dataframe["text"].tolist())
    all_labels = np.array(dataframe["label"].tolist())
    categories = set(all_labels)
    cat_cluster_map = {}
 
    all_embeddings = text_encoder.encode(all_texts)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    with open(output_dir+"stats/"+method+f"_embeddings.npz", 'wb') as file:
        np.save(file, all_embeddings)

  
    for cat in categories:
        labels = all_labels[all_labels==cat]
        embeddings = all_embeddings[all_labels==cat]
        texts = all_texts[all_labels==cat]
        nb_clusters = embeddings.shape[0]//k
        cat_cluster_map[cat] = nb_clusters
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        with open(output_dir+"stats/"+f"{cat}-{method}"+f"_{nb_clusters}-clusters_embeddings.npz", 'wb') as file:
            np.save(file, embeddings)

        model = KMeans(nb_clusters)
        kmeans = model.fit(embeddings)
        np.save(output_dir+"stats/"+f"{cat}-{method}"+f"_{nb_clusters}-clusters_centers", kmeans.cluster_centers_)

        clusters = kmeans.labels_
    
        data = {"text":[], "label":[], "cluster":[]}
        embedding_dict = {}
        for c in clusters:
            embedding_dict[int(c)] = []
        for i in range(len(texts)):
            data["text"].append(texts[i])
            data["label"].append(labels[i])
            data["cluster"].append(clusters[i])
            embedding_dict[int(clusters[i])].append(embeddings[i].tolist())
            
        df = pd.DataFrame(data)#, index=dataframe.index)
        data_gb = df.groupby(by=["cluster"]).groups
        
        cluster_medoid_indices = []

        for cluster_id  in set(clusters):
            cluster = data_gb[cluster_id]
            cluster_embeddings = embedding_dict[int(cluster_id)]
            dists = euclidean_distances(cluster_embeddings, cluster_embeddings)
            medoid_indx = np.argmin(dists.sum(axis=0))
            #medoid = embeddings[medoid_indx]
            cluster_medoid_indices.append(cluster[medoid_indx])
        
    
        np.save(output_dir+"stats/"+f"{cat}-{method}"+f"_{nb_clusters}-clusters_medoid_indices", cluster_medoid_indices)
        df.to_csv(output_dir+"clusters/"+f"{cat}-{method}"+f"_{nb_clusters}-clusters.csv")
    
    return cat_cluster_map



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model checkpoint used for embeddings the sentences.")
    parser.add_argument("--method", type=str, help="method used (distilled, or adapted)")
    parser.add_argument("--data_dir", type=str, help="path to the data .csv file")
    parser.add_argument("--meta_data_dir", type=str, help="path to the meta data .csv file containing name of classes in the data")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--avg_cluster_size", type=int, help="average size f clusters")
    args = parser.parse_args()

    device = torch.device('cuda')

    class_names = pd.read_csv(args.meta_data_dir)['label'].tolist()
    dataframe = pd.read_csv(args.data_dir, sep=',', header=0)
    label2id = {}
    cls_id = 0
    for cls in class_names:
        label2id[cls] = cls_id
        cls_id += 1

    model_path =  args.model_path 
    bi_encoder = SentenceTransformer(model_path, device=device)   


    kmeans_cluster(bi_encoder, dataframe, args.avg_cluster_size, args.output_dir, method=args.method)

if __name__ == "__main__":
    main()