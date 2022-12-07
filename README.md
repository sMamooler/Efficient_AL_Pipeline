# Efficient Active Learning Pipeline for Legal Text Classification


## Installation
```
cd lrtc_lib
conda create --name <env_name> --file requirements.txt
conda activate <env_name> 
```

## Datasets
The Contract_NLI and LEDGAR datasets can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1PWnFtIaZcmIVlSAAE3P4vbGKDqRkB-M7), and prepared following the instructions in [Low-Resource Text Classification Framework](https://github.com/IBM/low-resource-text-classification-framework#adding-a-new-dataset).

## Task adaptaton Knowledge Distillation
A separate conda environment is needed for the task adaptation and knowledge distillation steps. 
```
cd task_adapt_and_knowledge_distill
conda create --name <env_name> --file requirements.txt
conda activate <env_name> 
```
Then the following commands can be used respectively:
```
bash run_task_adaptation.bash
bash run_knowedge_distillation.bash
```
Note that the .bash scripts need to be modified according to the dataset.

## Clustering
After the pre-trained model is adapted and distilled, the clusters can be computed using 
```
cd clustering
bash run_clustering.bash
```
Note that the bash script should be adapted to the model and dataset used. 


## Train with Active Learning
The following command can be used to fine-tune a model with active leanring:
```
python -m experiment_runners.experiment_runner_imbalanced --config <Path_to_config>
```
The config should at least contain the 
* Experiment name
* Dataset name
* Number of times to repeat the experiment
* Name of the categories in the dataset to be used
* Path to the checkpoint of the model used for fine-tuining

More parameters are needed for experiments using cluster medoids (explained in Sec 4.3 of the paper). 
Examples of config files can be found in configs.


## Plots
Once the active learning iterations are done, and the results are stored in csv format, you can run the following command to plot the results and visually compare different models and active learning strategies. The plot_reslts.py script should be adapted to the paths containing the experiment results. 

```
python experiment_runners/experiment_runners_core/plot_results.py
```

This script contains required function to produce figures 1 and 2 in the paper comparing different langauge modes and active learning startegies together. 


## Acknowledgment
This repository has been build up on [IBM's Low-Resource Text Classification Framework](https://github.com/IBM/low-resource-text-classification-framework) and borrows code from [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), and [Huggingface Transformers](https://github.com/huggingface/transformers).

## Reference
If you find this project useful, please consider citing it in your work :)
```
@article{Mamooler2022AnEA,
  title={An Efficient Active Learning Pipeline for Legal Text Classification},
  author={Sepideh Mamooler and R{\'e}mi Lebret and St{\'e}phane Massonnet and Karl Aberer},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.08112}
}
```



