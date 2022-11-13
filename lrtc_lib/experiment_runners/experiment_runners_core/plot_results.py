# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_palette("tab10")

def AL_comparison_plot(dataset, metric, dfs_dict, fully_supervised, output_path, avg=""):
    """
    Plots the f1 score for annotated dataset size from 10 to 60, and compares different active learning strategies. (Sec 6.1, Figure 2 of the paper)
    """

    methods = list(dfs_dict.keys())
    models = sorted(dfs_dict[methods[0]]["model"+avg].unique())
    als = sorted(dfs_dict[methods[0]]["AL"+avg].unique())
    categories = sorted(dfs_dict[methods[0]]["category"+avg].unique())
    
    if dataset.startswith("contract"):
        dataset = dataset.capitalize().replace("_nli", "-NLI")[:-5]
        ymin = 0.0
    else:
        dataset = dataset.upper()[:-5]
        ymin = 0.4

    x_col =  "train total count"+avg
    f1_scores = []
   
    # plot for each class
    line_labels = []
    all_dfs = pd.DataFrame({})
    for df_name, df in dfs_dict.items():
        df["method"] = [df_name]*len(df["AL"+avg])
        all_dfs = pd.concat([all_dfs, df])
        line_labels.append(f'{df_name}')

    line_labels = sorted(line_labels)
    all_dfs = all_dfs[all_dfs["AL"+avg]!="no_active_learning"]

    for cat in categories:

        cat_name = cat.replace("/", "").replace(" ", "_")
        fully_supervised_df = pd.read_csv(fully_supervised+cat.replace("/", "").replace(" ", "")+"test_res.csv")
       
        f1_scores.append(fully_supervised_df["test_f1-score"].values[0])
       
        for model in models:
            

            axis = []

            model_df_all = all_dfs[(all_dfs["model"+avg] == model) & (all_dfs["category"+avg] == cat)]
            model_df = model_df_all[[metric, "method", "AL"+avg]]
            if avg=="":
                model_df[x_col] = [20, 30, 40, 50, 60]*(len(model_df)//5)
            else:
                model_df[x_col] = [20, 30, 40, 50, 60]*(len(model_df)//5)

            model_df = model_df.dropna(axis=0)
            model_df = model_df.rename(columns = {"AL"+avg: "AL strategy"})
            ax = sns.lineplot(data=model_df, x=x_col, y=metric, hue="AL strategy", style="AL strategy", palette="flare", dashes=False, markers=True, legend=True)#, label=f'{df_name}')
            ax.set_xticks([20, 30, 40, 50, 60])
            ax.set_xlabel("Annotated dataset size", fontsize = 20.0)
            axis.append(ax)
            ax.set_ylabel("F1-score", fontsize = 20.0)
            ax.set_title(f"{dataset}", fontsize = 20.0)

        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title="")
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(output_path, f"{dataset}_{cat_name}.pdf"), dpi=1200)
        plt.savefig(os.path.join(output_path, f"{dataset}_{cat_name}.png"))
        plt.clf()

    # plot for each model
    line_labels = []
    all_dfs = pd.DataFrame({})
    for df_name, df in dfs_dict.items():
        df["method"] = [df_name]*len(df["AL"+avg])
        all_dfs = pd.concat([all_dfs, df])

    all_dfs = all_dfs[all_dfs["AL"+avg]!="no_active_learning"]
        
    line_labels = sorted(line_labels)
    for model in models:
        
        axis = []

        model_df_all = all_dfs[(all_dfs["model"+avg] == model)]
        model_df = model_df_all[[metric, "category"+avg, "AL"+avg]]
        
        if avg=="":
            model_df[x_col] = [20, 30, 40, 50, 60]*(len(model_df)//5)
        else:
            model_df[x_col] = [20, 30, 40, 50, 60]*(len(model_df)//5)

        
        model_df = model_df.dropna(axis=0)
        model_df = model_df.groupby([x_col, "AL"+avg], as_index=False).mean()
        model_df = model_df.rename(columns = {"AL"+avg: "AL strategy"})
        ax = sns.lineplot(data=model_df, x=x_col, y=metric, hue="AL strategy", style="AL strategy", palette="flare", dashes=False, markers=True, legend=True)#, label=f'{df_name}')
  

        ax.set_xticks([20, 30, 40, 50, 60])
        ax.set_xlabel("Annotated dataset size", fontsize = 20.0)
        ax.set_ylabel("F1-score", fontsize = 20.0)
        axis.append(ax)
        ax.set_title(f"{dataset}", fontsize = 20.0)
   
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="")
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(output_path, f"{dataset}_avg.pdf"), dpi=1200)
    plt.savefig(os.path.join(output_path, f"{dataset}_avg.png"))
    # clear figure
    plt.clf()
    


def method_comparison_plot(dataset, metric, dfs_dict, fully_supervised, output_path, avg=""):
    """
    Plots the f1 score for annotated dataset size from 10 to 60, and compares different models. (Sec 6.1, Figure 1 of the paper)
    """

    methods = list(dfs_dict.keys())
    models = sorted(dfs_dict[methods[1]]["model"+avg].unique())
    als = sorted(dfs_dict[methods[1]]["AL"+avg].unique())
    categories = sorted(dfs_dict[methods[1]]["category"+avg].unique())

    if dataset.startswith("contract"):
        dataset = dataset.capitalize().replace("_nli", "-NLI")[:-5]
        ymin = 0.0
    else:
        dataset = dataset.upper()[:-5]
        ymin = 0.5
    
    
    x_col =  "train total count"+avg
    f1_scores = []
   
    # plot for each class
    line_labels = []
    all_dfs = pd.DataFrame({})
    for df_name, df in dfs_dict.items():
        df["method"] = [df_name]*len(df["AL"+avg])
        all_dfs = pd.concat([all_dfs, df])
        line_labels.append(f'{df_name}')

    line_labels = sorted(line_labels)

    for cat in categories:

        cat_name = cat.replace("/", "").replace(" ", "_")
        fully_supervised_df = pd.read_csv(fully_supervised+cat.replace("/", "").replace(" ", "")+"test_res.csv")
        f1_scores.append(fully_supervised_df["test_f1-score"].values[0])
      
        for model in models:
            for al in als:
                
                if al == "no_active_learning":
                    continue

                axis = []
              
                model_df_all = all_dfs[(all_dfs["model"+avg] == model) & ((all_dfs["AL"+avg] == al) | (all_dfs["AL"+avg] == "no_active_learning")) & (all_dfs["category"+avg] == cat)]
                model_df = model_df_all[[metric, "method"]]
                if avg=="":
                    model_df[x_col] = [10, 20, 30, 40, 50, 60]*(len(model_df)//6)
                else:
                    model_df[x_col] = [10, 20, 30, 40, 50, 60]*(len(model_df)//6)

                model_df = model_df.dropna(axis=0).sort_values(["method",x_col])
                
                ax = sns.lineplot(data=model_df, x=x_col, y=metric, hue="method", style="method", palette="viridis", dashes=False, markers=True, legend=False)
                ax.set_ylabel("F1-score", fontsize = 20.0)
                ax.set_xticks([10, 20, 30, 40, 50, 60])
                ax.set_xlabel("Annotated dataset size", fontsize = 20.0)
                ax.set_ylim(ymin=ymin)
                axis.append(ax)

                if al=="DAL":
                    ax.set_title(f"{dataset}", fontsize = 20.0)
                else:
                    ax.set_title(f"{al}", fontsize = 20.0)
                
                plt.legend(
                    axis,     # The line objects
                    labels=line_labels,   # The labels for each line
                    loc="lower left",   # Position of legend
                    borderaxespad=0.1,    # Small spacing around legend box
                    fontsize = 11.0,
                )

                plt.subplots_adjust(bottom=0.2)
                plt.savefig(os.path.join(output_path, f"{dataset}_{cat_name}_{al}.pdf"), dpi=1200)
                plt.savefig(os.path.join(output_path, f"{dataset}_{cat_name}_{al}.png"))
                # clear figure
                plt.clf()
 
    # plot the average over all classes
    line_labels = []
    all_dfs = pd.DataFrame({})
    for df_name, df in dfs_dict.items():
        df["method"] = [df_name]*len(df["AL"+avg])
        all_dfs = pd.concat([all_dfs, df])
        line_labels.append(f'{df_name}')
        
    line_labels = sorted(line_labels)
    for model in models:
        for al_idx, al in enumerate(als):
            if al == "no_active_learning":
                continue
            
            axis = []
            
            model_df_all = all_dfs[(all_dfs["model"+avg] == model) & ((all_dfs["AL"+avg] == al) | (all_dfs["AL"+avg] == "no_active_learning"))]
            model_df = model_df_all[[metric, "method"]]
            
            if avg=="":
                model_df[x_col] = [10, 20, 30, 40, 50, 60]*(len(model_df)//6)
            else:
                model_df[x_col] = [10, 20, 30, 40, 50, 60]*(len(model_df)//6)

          
            model_df = model_df.dropna(axis=0)
            model_df = model_df.groupby([x_col,"method"], as_index=False).mean().sort_values(["method",x_col])
            ax = sns.lineplot(data=model_df, x=x_col, y=metric, hue="method", style="method", palette="viridis", dashes=False, markers=True, legend=False)#, label=f'{df_name}')
            ax.set_ylabel("F1-score", fontsize = 20.0)
            ax.set_xticks([10, 20, 30, 40, 50, 60])
            ax.set_xlabel("Annotated dataset size", fontsize = 20.0)
            ax.set_ylim(ymin=ymin)
            axis.append(ax)

            if al=="DAL":
                ax.set_title(f"{dataset}", fontsize = 20.0)
            else:
                ax.set_title(f"{al}", fontsize = 20.0)

            model_df.to_csv(os.path.join(output_path, f"{dataset}_avg_{al}.csv"))

            plt.legend(
                axis,     # The line objects
                labels=line_labels,   # The labels for each line
                loc="lower left",   # Position of legend
                borderaxespad=1,    # Small spacing around legend box
                fontsize = 11.0,
            )

            plt.subplots_adjust(bottom=0.2)
            
            plt.savefig(os.path.join(output_path, f"{dataset}_{al}_avg.pdf"), dpi=1200)
            plt.savefig(os.path.join(output_path, f"{dataset}_{al}_avg.png"))
            # clear figure
            plt.clf()


def plot_results(paths_dict, fully_supervised_file_path, rel_output_path, avg="", compare_ALs=False):
    dfs_dict = {}
    for path_name, path in paths_dict.items():
        df = pd.read_csv(path)
        dfs_dict[path_name] = df
      

    datasets = df["dataset"+avg].unique()
    metrics = ["f1"+avg]
   
    for metric in metrics:
        if not os.path.exists(rel_output_path):
            os.mkdir(rel_output_path)
        if compare_ALs:
            AL_comparison_plot(datasets[0], metric, dfs_dict, fully_supervised_file_path, rel_output_path, avg)
        else:
            method_comparison_plot(datasets[0], metric, dfs_dict, fully_supervised_file_path, rel_output_path, avg)


if __name__ == '__main__':

    # Contract_NLI
    ROOT_DIR = "output/experiments/paper_results/contract_nli/"
    fully_supervised_file_path = "output/fully_supervised/contract_nli/"
  
    contract_nli_baseline_path = "contract_nli_baseline/results/contract_nli_baseline_3_repeats_avg.csv"
    contract_nli_baseline_legalbert_path = "contract_nli_baseline_legalbert/results/contract_nli_baseline_legalbert_3_repeats_avg.csv"
    contract_nli_distilled_path = "contract_nli_distilled/results/contract_nli_distilled_3_repeats_avg.csv"
    contract_nli_medoid_sampling_distilled_path = "contract_nli_medoid_sampling_distilled/results/contract_nli_medoid_sampling_distilled_3_repeats_avg.csv"
    contract_nli_adapted_path = "contract_nli_adapted/results/contract_nli_adapted_3_repeats_avg.csv"
    
    # compare models
    paths_dicts = {
        "PT RoBERTa" : os.path.join(ROOT_DIR, contract_nli_baseline_path),
        "LEGALBERT" : os.path.join(ROOT_DIR, contract_nli_baseline_legalbert_path),
        "TAPT RoBERTa" : os.path.join(ROOT_DIR, contract_nli_adapted_path),
        "DisTAPT RoBERTa" : os.path.join(ROOT_DIR, contract_nli_distilled_path),
        "DisTAPT RoBERTa with IS" : os.path.join(ROOT_DIR, contract_nli_medoid_sampling_distilled_path),
    }
    plot_results(paths_dicts, fully_supervised_file_path, ROOT_DIR+"/figures/sec1/", avg="_avg")

    # compare AL strategies
    paths_dicts = {
        "DisTAPT RoBERTa with IS" : os.path.join(ROOT_DIR, contract_nli_medoid_sampling_distilled_path),
    }
    plot_results(paths_dicts, fully_supervised_file_path, ROOT_DIR+"/figures/sec2/", avg="_avg", compare_ALs=True)
    

    # LEDGAR
    ROOT_DIR = "output/experiments/paper_results/ledgar/"
    fully_supervised_file_path = "output/fully_supervised/ledgar/"
    
    ledgar_baseline_path = "ledgar_baseline/results/ledgar_baseline_3_repeats_avg.csv"
    ledgar_baseline_legalbert_path = "ledgar_baseline_legalbert/results/ledgar_baseline_legalbert_3_repeats_avg.csv"
    ledgar_distilled_path = "ledgar_distilled/results/ledgar_distilled_3_repeats_avg.csv"
    ledgar_medoid_sampling_distilled_path = "ledgar_medoid_sampling_distilled/results/ledgar_medoid_sampling_distilled_3_repeats_avg.csv"
    ledgar_adapted_path = "ledgar_adapted/results/ledgar_adapted_3_repeats_avg.csv"

    # compare models
    paths_dicts = {
        "PT RoBERTa" : os.path.join(ROOT_DIR, ledgar_baseline_path),
        "LEGABERT" : os.path.join(ROOT_DIR, ledgar_baseline_legalbert_path),
        "TAPT RoBERTa" : os.path.join(ROOT_DIR, ledgar_adapted_path),
        "DisTAPT RoBERTa" : os.path.join(ROOT_DIR, ledgar_distilled_path),
        "DisTAPT RoBERTa with IS" : os.path.join(ROOT_DIR, ledgar_medoid_sampling_distilled_path)
    
    }
    plot_results(paths_dicts, fully_supervised_file_path, ROOT_DIR+"/figures/sec1/", avg="_avg")

    # compare AL strategies
    paths_dicts = {
        "DisTAPT RoBERTa with IS" : os.path.join(ROOT_DIR, ledgar_medoid_sampling_distilled_path)
    
    }
    plot_results(paths_dicts, fully_supervised_file_path, ROOT_DIR+"/figures/sec2/", avg="_avg", compare_ALs=True)