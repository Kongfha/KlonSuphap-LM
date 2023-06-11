import re
import json
import torch
from torch.utils.data import Dataset

class KlonSuphapDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, mask = False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask = mask
        special_toks = ["<s1>", "</s1>", "<es1>", "</es1>", "<s2>", "</s2>", "<es2>", "</es2>", "<s3>", "</s3>"]
        special_tok_ids = []
        for tok in special_toks:
            special_tok_ids.append(tokenizer.encode(tok)[0])
        self.special_tok_ids = special_tok_ids

        delim_toks = ["\t","\n"]
        delim_tok_ids = []
        for tok in delim_toks:
            delim_tok_ids.append(tokenizer.encode(tok)[0])
        self.delim_tok_ids = delim_tok_ids

        lines = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for klon in data:
            bahts = list(filter(None, re.split(r'(?<=\n)', klon)))
            line = ""
            for i in range(0,len(bahts),2):
                line += bahts[i] + bahts[i+1]
                if i % 8 == 6 or i == len(bahts) - 2: # Split to chunk of 4 bots (8 lines)
                    lines.append(line)
                    line = ""
                
        self.lines = lines

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        encoded = self.tokenizer.encode_plus(
            line,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        if self.mask:
            masking_state = False
            eos_id = self.tokenizer.encode(self.tokenizer.eos_token)[0]
            for i, ids in enumerate(input_ids):
                if int(ids) == eos_id:
                    break
                if int(ids) in self.special_tok_ids:
                    masking_state = not(masking_state)
                    continue
                if int(ids) in self.delim_tok_ids:
                    continue
                if not(masking_state):
                    attention_mask[i] = torch.tensor(0)
            
        return {'input_ids': input_ids, 'attention_mask': attention_mask}