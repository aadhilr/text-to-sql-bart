import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from config import read_arguments_train, read_arguments_evaluation
from training import train
from evaluation import evaluate
from transformers import BartForConditionalGeneration, BartTokenizer
from utils import setup_device, save_model, create_experiment_folder
from spider import spider_utils
import wandb
wandb.init(project="text_to_sql_bart")

if __name__ == '__main__':
    args = read_arguments_train()
    device, n_gpu = setup_device()

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    train_path = os.path.join(args.data_dir, "train.json")
    dev_path = os.path.join(args.data_dir, "dev.json")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    train_input_texts = [d['question'] for d in train_data]
    train_target_texts = [d['query'] for d in train_data]

    dev_input_texts = [d['question'] for d in dev_data]
    dev_target_texts = [d['query'] for d in dev_data]

    train_dataset = spider_utils.SpiderDataset(train_input_texts, train_target_texts, tokenizer)
    dev_dataset = spider_utils.SpiderDataset(dev_input_texts, dev_target_texts, tokenizer)

    batch_size = args.batch_size
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    args_e = read_arguments_evaluation()
    dev_data_loader = DataLoader(dev_dataset, batch_size=args_e.batch_size, shuffle=False)

    wandb.watch(model, log='parameters')
    epochs = args.num_epochs
    output_path = args.model_output_dir
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loss = train(model, tokenizer, train_data_loader, device)
        print(f"Epoch {epoch + 1}: loss = {loss}")
        acc = evaluate(model, tokenizer, dev_data_loader)

        if acc > best_acc:
            save_model(model, os.path.join(output_path))
            tqdm.write(
                "Accuracy of this epoch ({}) is higher then the so far best accuracy ({}). Save model.".format(acc,
                                                                                                               best_acc))
            best_acc = acc


