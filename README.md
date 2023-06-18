# 🌾 KlonSuphap-LM (แต่งกลอนแปด ด้วย GPT-2)

Visit Demo Space -> [Kongfha/KlonSuphap-Generator](https://huggingface.co/spaces/Kongfha/KlonSuphap-Generator) <br>
Visit Huggingface Model Card -> [Kongfha/KlonSuphap-LM](https://huggingface.co/Kongfha/KlonSuphap-LM) <br>
Visit Blog (Thai Language) -> [🌾 KlonSuphap-LM แต่งกลอนแปด ด้วย GPT-2](https://medium.com/@kampanatyingseree4704/klonsuphap-lm-%E0%B9%81%E0%B8%95%E0%B9%88%E0%B8%87%E0%B8%81%E0%B8%A5%E0%B8%AD%E0%B8%99%E0%B9%81%E0%B8%9B%E0%B8%94-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-gpt-2-d2baffc80907)

## Table of Contents
- [Introduction](#introductionIntroduction)
- [Training Process](#training-process)
- [Limitation](#limitation)
- [Usage](#usage)
    - [Rhyme-Tagging Data](#rhyme-tagging-data)
    - [Fine-Tuning without non-rhyme-related mask](#fine-tuning-without-non-rhyme-related-mask)
    - [Fine-Tuning with non-rhyme-related mask](#fine-tuning-with-non-rhyme-related-mask)
    - [Fine-Tuning using Reinforcement Learning](#fine-tuning-using-reinforcement-learning)
    - [Testing Model](#testing-model)
    - [Generate Klon-Paed using Pretrained Model](#generate-klon-paed-using-pretrained-model)

## Introduction
This repository contains the code for training KlonSuphap-LM, a language model specialized in generating Thai Klon-Paed Poems.

The goal of this project is to train a Language Model for Thai Klon-Paed Poem generation, with a focus on ensuring that the model can accurately generate poems with correct rhyming patterns without using additional pipelines.

## Training Process 

**KlonSuphap-LM** or GPT-2 for Thai poems (Klon-Paed Poem).
I use [GPT-2 base Thai](https://huggingface.co/flax-community/gpt2-base-thai) as a pre-trained model for fine-tuning exclusively
on Thai Klon-Paed Poem (กลอนแปด) retrieved from [Thai Literature Corpora (TLC)](https://attapol.github.io/tlc.html?fbclid=IwAR1UGV8hKGphwcuRCOCjJkVE4nC9yQ1_M_lFnxx9CLl9IzVKGK_mtbotQzU)  dataset.

Prior to my recent poem-generation model, [PhraAphaiManee-LM](https://huggingface.co/Kongfha/PhraAphaiManee-LM/), although the model can perform a
depiction of Thai Klon-Paed Poems, it still does not adhere to the rules of Thai Klon-Paed (ฉันทลักษณ์) in its generated output. To overcome this challenge I developed techniques that make the model to be more adhere to rules are as follows.

1. **Fine-Tuning dataset preprocessing.<br>**
   &ensp;&ensp;As I have a limited quantity of Thai Klon-Paed Poem or about 65770 lines (บาท), thus to succeed in the objective of making the model to be more adhere to rules,
   I developed a technique called ***"Rhyme Tagging"***. <br>
   &ensp;&ensp;***"Rhyme Tagging"*** performs tag insertion before and after words that are expected to rhyme with the other words based on Klon-Paed Rules. <br>
   <u>**Example**</u><br>
   >&ensp;&ensp;พอได้ยินเสียงระฆังข้างหลัง\<s2>เขา\</s2><br>เห็นผู้\<es2>เฒ่า\</es2>ออกจากชะวาก\<s2>ผา\</s2><br>สรรพางค์ร่างกายแก่ช\<es2>รา\</es2><br>แต่ผิว\<es2>หน้า\</es2>นั้นละม้ายคล้ายทา\<s3>รก\</s3>&ensp;&ensp;
   
   With ***"Rhyme Tagging"***, the potential loss of rhyme information due to an overwhelming flood of non-rhyme-related data can be mitigated. This approach aids the self-attention mechanism in extracting a greater amount of rhyme-related information, ensuring its preservation and relevance throughout the processing.

2. **Applying Attention-Mask while fine-tuning.<br>**
   &ensp;&ensp;Apart from performing a common fine-tuning process using the preprocessed dataset, I did fine-tune the model by applying Attention-Mask to non-rhyme-related words to the dataset as following visualization.<br>
   <u>**Visualized Example**</u><br>
   >&ensp;&ensp;------------------------------\<s2>เขา\</s2><br>-----\<es2>เฒ่า\</es2>--------------------\<s2>ผา\</s2><br>---------------------------\<es2>รา\</es2><br>------\<es2>หน้า\</es2>-----------------------\<s3>รก\</s3>&ensp;&ensp;

   By applying Attention-Mask while fine-tuning, the model can prioritize the extraction of information from both the rhyme-tags and their surrounding words without dropping positional information.
   This enhances the model's performance in subsequent stages of fine-tuning as if the model were constructing lookup table for rhyme-related words. 

3. **Performing Reinforcement Learning<br>**
   &ensp;&ensp;After the stage of Supervised Fine-Tuning, I perform Reinforcement Learning to the model using [voidful/TextRL](https://github.com/voidful/TextRL) by defining ***Klon-Paed Grader*** as a PPO Environment.<br>
   &ensp;&ensp;I perform Reinforcement Learning by randomly pick initial 2-5 syllables from the validation set as text inputs in an observation list, then I force the model to generate only 1 line (บาท) which has only 1 rhyme pair.<br>
   &ensp;&ensp;TextRL will repeatedly feed text inputs from the observation list to the model and calculate the reward using my ***Klon-Paed Grader***, then update the model's weights based on rewards it recieved.

## Limitation
The current training process uses \<s2> and \<es2> tags for inner rhyme (สัมผัสภายในบท) and \<s3> tag for outer rhyme (สัมผัสระหว่างบท). However, the model tends to prioritize learning inner rhyme, likely due to the higher quantity of \<s2> and \<es2> tags compared to \<s3> tags. The reinforcement learning method used in training focuses on generating a single line of a poem, resulting in only one pair of \<s2> and \<es2> tags, potentially overshadowing the significance of \<s3> tags in the model's learning.

# Setup
```bash
pip install -r requirement.txt
```

## Usage

#### Rhyme-Tagging Data
```bash
python3 tag_data.py --raw_path ./path/to/raw_text.json \
                    --save_path ./path/to/save
```

#### Fine-Tuning without non-rhyme-related mask 
```bash
python3 train.py --mask False \
                 --train_path ./path/to/train_data.json \
                 --val_path ./path/to/valid_data.json \
                 --tokenizer_path ./path/to/tokenizer \
                 --pretrained_path ./path/to/model \
                 --batch_size BATCH_SIZE \
                 --epochs NUMBER_OF_EPOCHS \
                 --save_path ./path/to/save
```

#### Fine-Tuning with non-rhyme-related mask 
```bash
python3 train.py --mask True \
                 --train_path ./path/to/train_data.json \
                 --val_path ./path/to/valid_data.json \
                 --tokenizer_path ./path/to/tokenizer \
                 --pretrained_path ./path/to/model \
                 --batch_size BATCH_SIZE \
                 --epochs NUMBER_OF_EPOCHS \
                 --save_path ./path/to/save
```

#### Fine-Tuning using Reinforcement Learning
```bash
# Train RL
python3 train_RL.py --observation_path ./path/to/observation_list.json \
                    --tokenizer_path  ./path/to/tokenizer \
                    --pretrained_path ./path/to/model \
                    --steps NUMBER_OF_STEPS \
                    --update_interval UPDATE_INTERVAL \
                    --minibatch_size MINI_BATCH_SIZE \
                    --epochs NUMBER_OF_EPOCHS \
                    --save_path ./path/to/save 
                    # saved model will be ./path/to/save/{step}_finish and ./path/to/save/best

# Dump saved model to Huggingface format
python3 dump_RL.py  --model ./path/to/model \
                    --tokenizer ./path/to/tokenizer \
                    --rl ./path/to/save/best \
                    --dumpdir ./path/to/save/dumped_model
```

#### Testing Model
```bash
python3 test.py  --input_path ./path/to/test_inputs.json \
                 --model_path ./path/to/model \
                 --tokenizer_path ./path/to/tokenizer \
                 --max_length MAX_LENGTH \
                 --top_p TOP_P \
                 --temperature TEMPERATURE \
                 --save_path ./path/to/test_result
```

#### Rhymes Evaluation
```bash
python3 sumpass_eval.py  --test_path ./path/to/test_result.json \
                         --eval_save_path ./path/to/save 
```

#### Generate Klon-Paed using Pretrained Model
```python
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Kongfha/KlonSuphap-LM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generate = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer)

input_sentence = "มิตรแท้"
generated_text = generate(input_sentence,
                          max_length=160,
                          top_p=0.85,
                          temperature=1)
# generation parameters can be varied 

print(f"Input: {input_sentence}")
print(f"Output:\n {generated_text[0]['generated_text']}")
```

