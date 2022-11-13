import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import datasets
from transformers.trainer_callback import EarlyStoppingCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support



class Binary_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.BCEWithLogitsLoss()
        comp = 1-labels
        # [negative, positive]
        labels = torch.stack((comp, labels), dim=1)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        
        return (loss, outputs) if return_outputs else loss

class Legal_Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def read_legal_data(data_frame, class_name, output_dir, test=False):
    
    if "policy_id" in data_frame.columns:
        data_frame = data_frame.drop(columns=["policy_id"])
    if "segment_id" in data_frame.columns:
        data_frame = data_frame.drop(columns=["segment_id"])
        
    data_frame = data_frame.replace({class_name:1})
    data_frame.label[data_frame.label!=1]=0
  
    pos_data = data_frame[data_frame.label==1]
    neg_data = data_frame.drop(pos_data.index, axis=0,inplace=False)
    
    if not test:
        neg_data = neg_data.sample(n=len(pos_data), random_state=2, replace=False)
    
    data_frame = pd.concat([pos_data, neg_data])
    data_frame.sort_index(inplace=True) 

    if test:
        file_name = "binary_test_data.csv"
    else:
        file_name = "binary_train_data.csv"

    data_frame["label"] = data_frame["label"].astype(np.int64)
    data_frame.to_csv(output_dir+file_name)
   
    label2id = {'positive':1, 'negative':0}
    id2label = {1: 'positive', 0:'negative'}
   
    return file_name, label2id, id2label



def compute_metrics(eval_res):

    logits, labels = eval_res
    labels = torch.tensor(labels, dtype=torch.uint8).cpu()
    preds = torch.tensor(logits, dtype=torch.float)
    preds = F.softmax(preds, dim=1)

    preds = torch.argmax(preds, dim=1).cpu()

    prc, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
   
    return {
        'precision': prc,
        'recall': rec,
        'f1-score': f1
    }


def main():
    
    root_dir = "/mnt/nlp4sd/mamooler/low-resource-text-classification-framework/lrtc_lib/data/available_datasets/"
    
    dataset_name = "contract_nli"

    train_data_dir = "train.csv"
    test_data_dir = "test.csv" 
    dev_data_dir = "dev.csv"

    

    train_data_frame = pd.read_csv(os.path.join(root_dir, dataset_name, train_data_dir), sep=',', header=0)
    test_data_frame = pd.read_csv(os.path.join(root_dir, dataset_name, test_data_dir), sep=',', header=0)
    dev_data_frame = pd.read_csv(os.path.join(root_dir, dataset_name, dev_data_dir), sep=',', header=0)
    

    meta_data = train_data_frame.groupby(["label"]).size().reset_index(name='counts')
    class_names = meta_data['label'].tolist()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_train_epochs=50
    warmup_steps=10
    test_batch_size = 16
    train_batch_size = 8
    load_ckpt = False
    patience=10

    for cls_name in class_names:

        
        o_name = cls_name.replace('/', '_').replace(' ', '_')
        output_dir = os.path.join('/mnt/nlp4sd/mamooler/low-resource-text-classification-framework/lrtc_lib/output/fully_supervised/legal_bert/', dataset_name, cls_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        train_file, label2id, id2label = read_legal_data(train_data_frame, cls_name, output_dir)
        test_file, _, _ = read_legal_data(test_data_frame, cls_name, output_dir, test=True)
        dev_file, _, _ = read_legal_data(dev_data_frame, cls_name, output_dir, test=True)

        validation_dataset = datasets.load_dataset('csv', data_files=output_dir+dev_file, split='train')
        train_dataset = datasets.load_dataset('csv', data_files=output_dir+train_file, split='train')
        test_dataset = datasets.load_dataset('csv', data_files=output_dir+test_file, split='train')
        

        training_args = TrainingArguments(
                output_dir = output_dir+"output",
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size = train_batch_size,
                gradient_accumulation_steps = 16,    
                per_device_eval_batch_size= test_batch_size,
                do_train = True,
                do_eval = True,
                evaluation_strategy = "epoch",
                metric_for_best_model = "f1-score",
                save_strategy = "epoch",
                disable_tqdm = False, 
                load_best_model_at_end=True,
                save_total_limit=1,
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                logging_steps = 10,
                fp16 = True,
                logging_dir=output_dir+"logs",
                dataloader_num_workers = 8,
                run_name = f'binary-classification-{o_name}'
            )

        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased") # 'roberta-base'
        def tokenize_function(examples):
            segments = list(examples["text"])
            return tokenizer(segments, truncation=True, padding='max_length')


    
        validation_dataset = validation_dataset.map(tokenize_function, batched=True)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        # config = AutoConfig.from_pretrained("nlpaueb/legal-bert-base-uncased") # 'roberta-base'
        # config.label2id = label2id
        # config.id2label = id2label

        # model = RobertaForSequenceClassification(config).to(device)
        model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # model= nn.DataParallel(model)
        # model.to(device)
        #model = RobertaForSequenceClassification.from_pretrained(output_dir+"output/checkpoint-168")
        
        trainer = Binary_Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )

        early_stopping_callback = EarlyStoppingCallback(patience)
        trainer.add_callback(early_stopping_callback)

        trainer.train()
       
        _,_, test_res = trainer.predict(test_dataset, metric_key_prefix="test")
        

        test_res = pd.DataFrame(test_res, index=[0])
        test_res.to_csv(output_dir+f"test_res.csv")

if __name__== "__main__" :
    main()