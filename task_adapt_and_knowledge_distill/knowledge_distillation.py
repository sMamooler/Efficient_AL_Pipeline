# adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/distillation/model_distillation.py 

"""
This file contains an example how to make a SentenceTransformer model faster and lighter.

This is achieved by using Knowledge Distillation: We use a well working teacher model to train
a fast and light student model. The student model learns to imitate the produced
sentence embeddings from the teacher. We train this on a diverse set of sentences we got
from MAPS dataset.
"""
from ast import arg
from sys import stderr
import logging
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, evaluation
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.datasets import ParallelSentencesDataset
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, help="name or path of the teacher model", default='stsb-roberta-base-v2')
    parser.add_argument("--student_model", type=str, help="name or path of the student model")
    parser.add_argument("--output_path", type=str, help="path to directory where checkpoints will be stored")
    parser.add_argument("--dataset_path", type=str, help="path to the dataset .txt file")
    parser.add_argument("--nb_epochs", type=int, help="number of epochs for training", default=10)
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    # Teacher Model: pre-trained bi-encoder
    teacher_model_name = args.teacher_model #'stsb-roberta-base-v2'
    teacher_model = SentenceTransformer(teacher_model_name)

    print("# prams: ",sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))
    # Student Model: fine-tuned RoBERTa
    student_model_path = args.student_model #"/mnt/nlp4sd/mamooler/checkpoints/domain_adaptation/res/ledgar/"
    word_embedding_model = models.Transformer(student_model_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    output_path = "/".join([args.output_path, f"model-distillation-{args.nb_epochs}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")])  #"/mnt/nlp4sd/mamooler/checkpoints/domain_adaptation/output/ledgar/model-distillation-10epochs-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_path = args.dataset_path #"/mnt/nlp4sd/mamooler/low-resource-text-classification-framework/lrtc_lib/data/available_datasets/ledgar/LEDGAR.txt" 

    inference_batch_size = 64
    train_batch_size = 32



    with open(dataset_path, 'r', encoding='utf8') as fIn:
        sentences = fIn.readlines()
        
    train_sentences, dev_sentences= train_test_split(sentences, train_size=0.9, random_state=42)

    train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=False)
    train_data.add_dataset([[sent] for sent in train_sentences], max_sentence_length=256)


    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MSELoss(model=student_model)

    # We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
    dev_evaluator_mse = evaluation.MSEEvaluator(dev_sentences, dev_sentences, teacher_model=teacher_model)


    logging.info("Start model distillation")
    # Train the student model to imitate the teacher
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator= dev_evaluator_mse,
                    epochs=args.nb_epochs,
                    warmup_steps=10000,
                    evaluation_steps=500000,
                    output_path=output_path,
                    save_best_model=True,
                    optimizer_params={'lr': 1e-4, 'eps': 1e-6, 'correct_bias': False},
                    use_amp=True)

if __name__ == "__main__":
    main()
