from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm


def train(model, tokenizer, train_data_loader, device):
    model.to(device)
    model.train()
    train_loss = 0.0
    optimizer = AdamW(model.parameters(), lr=1e-5)
    for batch in tqdm(train_data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_mask = batch['target_mask'].to(device)
        decoder_input_ids = target_ids[:, :-1].contiguous()
        labels = target_ids[:, 1:].clone()
        # ignore padding tokens
        labels[target_mask[:, 1:] == 0] = -100
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                            labels=labels)
        loss = outputs.loss
        # total_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        avg_loss = train_loss / len(train_data_loader)
        # print(f'Training loss: {avg_loss:.4f}')
    return avg_loss