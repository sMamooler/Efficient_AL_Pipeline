import json
import pandas as pd
from sklearn.model_selection import train_test_split


input_dir = "./available_datasets/ledgar/LEDGAR_2016-2019_clean.jsonl"

text_file = open("./available_datasets/ledgar/LEDGAR.txt", "wb")

with open(input_dir, 'r') as file:
    json_list = list(file)


data_dict = {"text":[], "label":[]}
for json_str in json_list:
    line = json.loads(json_str)
    for l in line["label"]:
        data_dict["text"].append(line["provision"])
        data_dict["label"].append(l)

df = pd.DataFrame(data_dict, columns=["text", "label"])
df_freq = df.groupby("label").size().reset_index(name='count').set_index("label")

frequencies = df_freq["count"].values
print(max(frequencies))
print(len(df))

freq_records = {"text":[], "label":[]}
for i,label in enumerate(df["label"]):
    if df_freq.loc[label]['count']>=10000:
        freq_records["text"].append(df["text"][i])
        freq_records["label"].append(label)
        text_file.write((df["text"][i]+'\n').encode('ascii',errors='ignore'))


text_file.close()

data = pd.DataFrame(freq_records, columns=["text", "label"])

train, test = train_test_split(data, test_size=0.2)
train, dev = train_test_split(train, test_size=1/7)

train.to_csv("./available_datasets/ledgar/train.csv")
test.to_csv("./available_datasets/ledgar/test.csv", index=False)
dev.to_csv("./available_datasets/ledgar/dev.csv", index=False)

stat = data.groupby("label").size().reset_index(name='count')
train_stat = train.groupby("label").size().reset_index(name='count')
test_stat = test.groupby("label").size().reset_index(name='count')
dev_stat = dev.groupby("label").size().reset_index(name='count')


stat.to_csv("./available_datasets/ledgar/stat.csv", index=False)
train_stat.to_csv("./available_datasets/ledgar/train_stat.csv", index=False)
test_stat.to_csv("./available_datasets/ledgar/test_stat.csv", index=False)
dev_stat.to_csv("./available_datasets/ledgar/dev_stat.csv", index=False)
