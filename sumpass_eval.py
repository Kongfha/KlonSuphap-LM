import json
import pandas as pd
from tqdm import tqdm

from utils.word_check import replace_long_short, format_str_waks, format_waks_syl

import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--test_path', metavar='TEST-PATH', type=str, required=True, help='path to test outputs')
parser.add_argument('--eval_save_path', metavar='SAVE-PATH', type=str, default = "./model", help='path to evaluation result directory')

args = parser.parse_args()

def sumpass_score(bot_VowMat,bot_th): # bot_VowMat = ['a', 'xxN', 'oot', '@@t']
    #score = [s1,s2_1,s2_2,s2_3] สดับ-รับ รับ-รอง รับ-ส่ง รอง-ส่ง
    score = [0, 0, 0, 0]
    repli = [0, 0, 0, 0]

    s1 = replace_long_short(bot_VowMat[0][-1])
    s1_th = bot_th[0][-1]

    for i in [1,2,3,4]:
        cur = replace_long_short(bot_VowMat[1][i])
        cur_th = bot_th[1][i]
        if s1 == cur:
            score[0] += 1
            if s1_th == cur_th:
                repli[0] += 1
            break

    s2 = replace_long_short(bot_VowMat[1][-1])
    s2_th = bot_th[1][-1]

    if s2 == replace_long_short(bot_VowMat[2][-1]):
        score[1] += 1
        if s2_th == bot_th[2][-1]:
            repli[1] += 1

    for i in [1,2,3,4]:
        cur = replace_long_short(bot_VowMat[3][i])
        cur_th = bot_th[3][i]
        if s2 == cur:
            score[2] += 1
            if s2_th == cur_th:
                repli[2] += 1
            break
    
    s2_2 = replace_long_short(bot_VowMat[2][-1])
    s2_2_th = bot_th[2][-1]
    for i in [1,2,3,4]:
        cur = replace_long_short(bot_VowMat[3][i])
        cur_th = bot_th[3][i]
        if s2_2 == cur:
            score[3] += 1
            if s2_2_th == cur_th:
                repli[3] += 1
            break

    return score, repli

def remove_excess_elements(lst, limit):
    if len(lst) > limit:
        lst = lst[:limit]
    return lst

def get_score(txt):
    #score = [s1,s2_1,s2_2,s2_3] สดับ-รับ รับ-รอง รับ-ส่ง รอง-ส่ง

    txt_waks = format_str_waks(txt)    
    klon_VowMat, klon_th, fail = format_waks_syl(txt_waks)

    bots_VowMat = [klon_VowMat[0:4], klon_VowMat[4:8]]
    bots_th = [klon_th[0:4], klon_th[4:8]]

    if fail:
        return "WordFail"
    
    for wak in klon_th:
        if len(wak) < 5 or len(wak) > 10:
            return "LengthFail"
    
    score = []
    repli = []
    for i in range(len(bots_VowMat)):
        cur_score, cur_repli = sumpass_score(bot_VowMat = bots_VowMat[i], bot_th=bots_th[i])
        score+=cur_score
        repli+=cur_repli
    
    s3 = replace_long_short(bots_VowMat[0][3][-1])
    s3_th = bots_th[0][3][-1]
    es3 = replace_long_short(bots_VowMat[1][1][-1])
    es3_th = bots_th[1][1][-1]

    score.append(0)
    repli.append(0)
    if s3 == es3:
        score[8] += 1
        if s3_th == es3_th:
            repli[8] += 1

    return [score, repli]

test_path = args.test_path
print(f"Loading test result from {test_path}")
with open(test_path) as f:
    data = json.load(f)

if __name__ == "__main__":
    list_df = []
    for klon in tqdm(data):
        cur_row = [klon["input"],klon["output"]]
        fail_count = [0,0]
        score = [0,0,0,0,0,0,0,0,0]
        repli = [0,0,0,0,0,0,0,0,0]

        result = get_score(klon["output"])  

        if result == "WordFail":
            fail_count[0] += 1
            cur_row.extend(fail_count+score+repli)
            list_df.append(cur_row)
            continue
            
        elif result == "LengthFail":
            fail_count[1] += 1
            cur_row.extend(fail_count+score+repli)
            list_df.append(cur_row)
            continue

        else:
            cur_row.extend(fail_count+result[0]+result[1])
            list_df.append(cur_row)

    columns = ["input", "output", "WordFail", "LenghtFail", "สดับ1-รับ1", "รับ1-รอง1", "รับ1-ส่ง1", "รอง1-ส่ง1", "สดับ2-รับ2", "รับ2-รอง2", "รับ2-ส่ง2", "รอง2-ส่ง2", "ส่ง1-รับ2","ซ้ำสดับ1-รับ1", "ซ้ำรับ1-รอง1", "ซ้ำรับ1-ส่ง1", "ซ้ำรอง1-ส่ง1", "ซ้ำสดับ2-รับ2", "ซ้ำรับ2-รอง2", "ซ้ำรับ2-ส่ง2", "ซ้ำรอง2-ส่ง2", "ซ้ำส่ง1-รับ2"]
    df = pd.DataFrame(list_df, columns = columns, dtype = int)
    
    save_path = args.eval_save_path

    save_eval_path = save_path+"/test_eval.csv"
    print(f"Saving evaluation result to {save_eval_path}")
    df.to_csv(save_eval_path)
    save_sum_eval_path = save_path+"/test_eval_summary.csv"
    print(f"Saving evaluation summary to {save_sum_eval_path}")
    summary_df = df.sum(axis=0) 
    summary_df.iloc[2:].to_csv(save_sum_eval_path)
    print("Finish")