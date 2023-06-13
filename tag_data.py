import json
from tqdm import tqdm
import os

from utils.word_check import Get_Vow_and_Syl, long_short_vow

import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--raw_path', metavar="RAW-PATH", type=str, required=True, help="raw dataset path")
parser.add_argument('--save_path', metavar="SAVE-PATH", type =str, required=True, help="tagged dataset save path")

args = parser.parse_args()

def tag_klon(txt):
    token_s1 = ["<s2>", "</s2>"] 
    token_es1 = ["<es2>", "</es2>"] # I changed <s1> <es1> to <s2> <es2> to increase quantity of tags 
    token_s2 = ["<s2>", "</s2>"]
    token_es2 = ["<es2>", "</es2>"]
    token_s3 = ["<s3>", "</s3>"]
    out = ""
    out_lst = []
    for i in range(0,len(txt),2):
        bot_raw = []
        bot_VowMat = []
        bot_th_syl = []
        bot_raw.extend(txt[i].split("\t"))
        bot_raw.extend(txt[i+1].split("\t"))
        have_fail = False
        for wak in bot_raw:
            VowMatList, th_syllable, fail = Get_Vow_and_Syl(wak)
            have_fail = have_fail or fail
            bot_VowMat.append(VowMatList)
            bot_th_syl.append(th_syllable)
        
        if have_fail:
            out_lst.append(out)
            out = ""
            continue
        
        wak1 = list(bot_th_syl[0])
        wak1.insert(-1, token_s1[0])
        wak1.append(token_s1[1])
        for word in wak1:
            out += word
        out += "\t"

        s1 = bot_VowMat[0][-1]
        for long_short in long_short_vow:
            s1 = s1.replace(long_short[0],long_short[1])

        wak2 = list(bot_th_syl[1])

        for ind, VowMat in enumerate(bot_VowMat[1]):
            if ind == 0:
                continue
            if ind >= 5:
                break
            for long_short in long_short_vow:
                VowMat = VowMat.replace(long_short[0],long_short[1])
            if VowMat == s1:
                wak2.insert(ind, token_es1[0])
                wak2.insert(ind+2, token_es1[1])
                break
        wak2.insert(-1,token_s2[0])
        wak2.append(token_s2[1])
        for word in wak2:
            out += word
        out += "\n"

        s2_1 = bot_VowMat[1][-1]
        for long_short in long_short_vow:
            s2_1 = s2_1.replace(long_short[0],long_short[1])

        wak3 = list(bot_th_syl[2])
        wak3.insert(-1, token_es2[0])
        wak3.append(token_es2[1])
        for word in wak3:
            out += word
        out += "\t"

        s2_2 = bot_VowMat[2][-1]
        for long_short in long_short_vow:
            s2_2 = s2_2.replace(long_short[0],long_short[1])
        wak4 = list(bot_th_syl[3])
        for ind, VowMat in enumerate(bot_VowMat[3]):
            if ind == 0:
                continue
            if ind >= 5:
                break
            for long_short in long_short_vow:
                VowMat = VowMat.replace(long_short[0],long_short[1])
            if VowMat == s2_1 or VowMat == s2_2:
                wak4.insert(ind, token_es2[0])
                wak4.insert(ind+2, token_es2[1])
                break
        wak4.insert(-1,token_s3[0])
        wak4.append(token_s3[1])
        for word in wak4:
            out += word
        out += "\n"

    if out != "":
        out_lst.append(out)            

    return out_lst

if __name__ == "__main__":
    load_path = args.raw_path
    print(f"Retrieving Data from {load_path}")
    with open(load_path) as f:
        data = json.load(f)

    tagged = []
    i = 0

    try:
        print("-----Tagging Data-----")
        for klon in tqdm(data):
            tagged.extend(tag_klon(klon))
            i += 1

        save_directory = args.save_path
        CHECK_FOLDER = os.path.isdir((save_directory))

        if not CHECK_FOLDER:
            os.makedirs((save_directory))
            print("created folder : ", (save_directory))
        else:
            print((save_directory), "folder already exists.")

        save_path = save_directory + f"/tagged_0to{i}elems_text.json"
        print(f"Saving Data to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(tagged, f, ensure_ascii=False, indent=2)

        print("Finished")
    except:
        print("Error Occurs")

        save_directory = args.save_path
        CHECK_FOLDER = os.path.isdir((save_directory))

        if not CHECK_FOLDER:
            os.makedirs((save_directory))
            print("created folder : ", (save_directory))
        else:
            print((save_directory), "folder already exists.")

        error_save_path = save_directory + f"/tagged_0to{i}elems_text.json"
        print(f"Saving Data to {error_save_path}")
        with open(error_save_path, 'w') as f:
            json.dump(tagged, f, ensure_ascii=False, indent=2)
        print("Finished")