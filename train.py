import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.my_loader import KlonSuphapDataset
from utils.train_functions import train, eval

import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--mask', metavar="MASK", type=bool, required=True, help="apply mask on non-rhyme-related data")
parser.add_argument('--train_path', metavar="TRAIN-PATH", type=str, required=True, help="training dataset path")
parser.add_argument('--val_path', metavar="VAL-PATH", type =str, required=True, help="validation dataset path")
parser.add_argument('--tokenizer_path', metavar="TOKEN-PATH", type =str, required=True, help="tokenizer path")
parser.add_argument('--pretrained_path', metavar="PRE-PATH", required=True, help="pretrained model path")
parser.add_argument('--batch_size',metavar='BATCH-SIZE', type=int, required=True, help="batch size")
parser.add_argument('--epochs',metavar='NUM-EPOCH', type=int, required=True, help="number of epochs")
parser.add_argument('--lr',metavar='LR', type = float, default=2e-5, help="learning rate")
parser.add_argument('--save_path',metavar='SAVE-PATH', type = str, default=".", help="saving path for model and tokenizer")

args = parser.parse_args()


torch.cuda.empty_cache()

print("-----Loading Materials-----")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrain_tokenizer_path = args.tokenizer_path
pretrain_model_path = args.pretrained_path

print(f"Loading tokenizer and model from \"{pretrain_model_path}\" and \"{pretrain_tokenizer_path}\"")
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(pretrain_model_path)
model.to(device)
    
train_data_path = args.train_path
print(f"Loading train data from \"{train_data_path}\"")
train_dataset = KlonSuphapDataset(train_data_path, tokenizer, max_length=600)

valid_data_path = args.val_path
print(f"Loading valid data from \"{valid_data_path}\"")
valid_dataset = KlonSuphapDataset(valid_data_path, tokenizer, max_length=600)

BATCH_SIZE = args.batch_size
print(f"Preparing data loader with batch size = {BATCH_SIZE}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)


if __name__ == "__main__":

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_epochs = args.epochs

    state = []

    print(f"-----Start training for {num_epochs} epochs-----")

    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, optimizer, tokenizer)
        
        valid_loss = eval(model, valid_loader, tokenizer)

        state.append({"epoch":epoch+1,"train_loss":train_loss,"valid_loss":valid_loss})
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    save_path = args.save_path

    model_save_path = save_path + f"/model_{num_epochs}_eps"
    print(f"Saving model to \"{model_save_path}\"")    
    model.save_pretrained(model_save_path)

    tokenizer_save_path = save_path + "/tokenizer_control2"
    print(f"Saving model to \"{tokenizer_save_path}\"")    
    tokenizer.save_pretrained(tokenizer_save_path)

    state_save_path = save_path + f"/state_{num_epochs}_eps.json"
    print(f"Saving states to \"{state_save_path}\"")
    with open(state_save_path, "w") as f:
        json.dump(state, f, indent=2)