import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, optimizer, tokenizer):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    return train_loss

def eval(model, valid_loader, tokenizer):
    model.eval()
    valid_loss = 0
    for batch in tqdm(valid_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        valid_loss += loss.item()
    valid_loss /= len(valid_loader)

    return valid_loss