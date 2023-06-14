from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--input_path', metavar='INPUT-PATH', type=str, required=True, help='path to input texts')
parser.add_argument('--model_path', metavar='MODEL-PATH', type=str, default = "./model", help='path to model')
parser.add_argument('--tokenizer_path', metavar='TOK-PATH', type=str, default = "./tokenizer", help='path to tokenizer')
parser.add_argument('--max_length', metavar='MAXLENGTH', type=int, default = 140, help = "max length of output")
parser.add_argument('--top_p', metavar='TOP-P', type=float, default = 0.8, help = "top_P")
parser.add_argument('--temperature', metavar='TEMP', type=float, default = 1.0, help = "temperature")
parser.add_argument('--save_path', metavar='SAVE-PATH', type=str, required=True, help='path to save outputs')


args = parser.parse_args()

pretrain_model_path = args.model_path
pretrain_tokenizer_path = args.tokenizer_path

print(f"Loading model and tokenizer from \"{pretrain_model_path}\" and \"{pretrain_tokenizer_path}\"")
model = AutoModelForCausalLM.from_pretrained(pretrain_model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_path = args.input_path
print(f"Loading input texts from \"{input_path}\"")
with open(input_path) as f:
    test_inputs = json.load(f)

if __name__ == "__main__":
    output_lst = []
    top_p = args.top_p
    temp = args.temperature
    max_length = args.max_length
    for input in tqdm(test_inputs):
        output = ""
        while len(output.split("\n")) <= 4:
            input_ids = tokenizer.encode(input)
            input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
            input_ids_tensor = input_ids_tensor.to(device)
            output = model.generate(input_ids_tensor, max_length = max_length,temperature=temp, top_p=top_p ,pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(output.reshape(-1))
            if(len(output.split("\n")) > 4):
                output = output[:output.find(output.split("\n")[4])]
        output_lst.append({"input":input,"output":output})
    
    save_path = args.save_path
    # I found that in colab when model.generate has been used the json.dump(ensure_ascii=False) will raised error
    try:
        file_save_path = save_path+"/test_result.json"
        print(f"Saving result to {file_save_path}")
        with open(file_save_path, "w") as f:
            json.dump(output_lst, f,ensure_ascii=False, indent=2)
    except:
        print("Error occurs")
        print("Changing encoding format to utf-8")
        file_save_path = save_path+"/test_result_utf8.json"
        print(f"Saving result to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as f:
            json.dump(output_lst, f,ensure_ascii=False, indent=2)
    
    print("Finished")


            