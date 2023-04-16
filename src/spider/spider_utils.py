from typing import List
import torch
from transformers import  BartTokenizer


class SpiderDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts: List[str], target_texts: List[str], tokenizer: BartTokenizer):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        input_tokens = self.tokenizer.batch_encode_plus([input_text], padding='max_length', truncation=True,
                                                        max_length=512)
        target_tokens = self.tokenizer.batch_encode_plus([target_text], padding='max_length', truncation=True,
                                                         max_length=512)
        input_ids = input_tokens['input_ids'][0]
        input_mask = input_tokens['attention_mask'][0]
        target_ids = target_tokens['input_ids'][0]
        target_mask = target_tokens['attention_mask'][0]
        return {'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(input_mask),
                'target_ids': torch.tensor(target_ids),
                'target_mask': torch.tensor(target_mask)}