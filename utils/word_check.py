from tltk import nlp
import re

pattern_1 = re.compile(r'\'|~|\|')
pattern_2 = re.compile(r'~|\|')
pattern_nt = re.compile(r'\t|\n')

long_short_vow = [['aa', 'a'],
                  ['ii', 'i'],
                  ['uu', 'u'],
                  ['ee', 'e'],
                  ['OO', 'O'],
                  ['xx', 'x'],
                  ['UU', 'U'],
                  ['uu', 'u'],
                  ['@@', '@'],
                  ['oo', 'o']]

special_tokens = ["<s1>", "</s1>", "<es1>", "</es1>", "<s2>", "</s2>", "<es2>", "</es2>", "<s3>", "</s3>"]
alphab = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y', 'z', '?', 'N']

def VowelMattra(txt):
    for i,c in enumerate(txt):
        if c not in alphab:
            txt = txt[i:]
            break
    VoMa = txt[:-1]
    return VoMa

def by_syllable(inp):
    for ch in ["๏", "๚ะ", "ฯ", "\n", "\t", " "]:
        inp = inp.replace(ch,"")
    text  = nlp.g2p(inp) 
    fail = False
    if text.count("<Fail>") > 0:
        fail = True
    txt_split = text.split('<tr/>')
    text_rom_2 = pattern_1.split(txt_split[1][:-5])
    text_rom_1 = pattern_2.split(txt_split[1][:-5])

    ind = []
    for i,rom in enumerate(text_rom_1):
        if rom.count('\''):
            ind.append(i)

    text_th_1 = pattern_1.split(txt_split[0])
    text_th_2 = []

    for i,word in enumerate(text_th_1):
        if i in ind:
            if(word[0] != 'เ' and word[0] != 'แ'):
                text_th_2.append(word[0])
                text_th_2.append(word[1:])
            else:
                text_th_2.append('')
                text_th_2.append(word)

        else:
            text_th_2.append(word)

    return text_rom_2, text_th_2, fail

def Get_Vow_and_Syl(txt):
    rom_syllable, th_syllable, fail = by_syllable(txt)
    VowMatList = [VowelMattra(a) for a in rom_syllable]
    return VowMatList, th_syllable, fail

def replace_long_short(str):
    for long_short in long_short_vow:
        str = str.replace(long_short[0],long_short[1])
    return str

def format_str_waks(txt):
    for token in special_tokens:
        txt = txt.replace(token,'')
    txt_waks = pattern_nt.split(txt)
    txt_waks = list(filter(lambda x: x != '', txt_waks))
    return txt_waks

def format_waks_syl(txt_waks):
    klon_VowMat = []
    klon_th = []

    fail = False

    for wak in txt_waks:
        VowMat, th_words, cur_fail = Get_Vow_and_Syl(wak)
        if cur_fail:
            fail = True
        klon_VowMat.append(VowMat)
        klon_th.append(th_words)

    return klon_VowMat, klon_th, fail
    

def sumpass_score(bot_VowMat,bot_th): # bot_VowMat = ['a', 'xxN', 'oot', '@@t']
    score = 0

    s1 = replace_long_short(bot_VowMat[0][-1])
    s1_th = bot_th[0][-1]

    brk = False
    for i in [1,2,3,4]:
        cur = replace_long_short(bot_VowMat[1][i])
        cur_th = bot_th[1][i]
        if s1 == cur and s1_th != cur_th:
            score += 20
            brk = True
            break
        elif s1 == cur:
            score += 5
            brk = True
            break
    if not(brk):
        score -= 20

    return score
