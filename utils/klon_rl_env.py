from utils.word_check import format_str_waks, format_waks_syl, sumpass_score
from textrl import TextRLEnv

def get_score(txt):
    global klon_len_error_count 
    global word_seg_fail_count 
    global less_than_two_wak_count 

    reward = 0
    txt_waks = format_str_waks(txt)

    if len(txt_waks) < 2:
        less_than_two_wak_count += 1
        reward -= 20
        return reward

    if len(txt_waks) > 2:
        txt_waks = txt_waks[:2]

    klon_VowMat, klon_th, fail = format_waks_syl(txt_waks)

    if fail:
        word_seg_fail_count += 1
        reward = 0
        return reward
    
    cnt = 0
    for wak in klon_th:
        if len(wak) < 5 or len(wak) > 10:
            klon_len_error_count += 1
            cnt += 1
    if cnt != 0:
        reward = -20 * cnt
        return reward

    
    reward += sumpass_score(bot_VowMat = klon_VowMat, bot_th=klon_th)

    return reward

class KlonRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        reward = 0
        if finish or len(predicted_list[0]) >= self.env_max_length:
            global reward_cnts
            global klon_len_error_count 
            global word_seg_fail_count 
            global less_than_two_wak_count 
            global reward_cnts
            global deque
            global tokenizer
            predicted_text = tokenizer.convert_tokens_to_string(predicted_list[0])
            all_text = input_item["input"]+predicted_text
            reward = get_score(all_text)
            deque.append(reward)
            reward_cnts += 1            
            if len(deque) < 50:
                print(f"{reward_cnts} | cur_reward: {reward} | len_error: {klon_len_error_count} | wak_error: {less_than_two_wak_count} | seg_fail : {word_seg_fail_count} |", end="\r")
            else:
                print(f"{reward_cnts} | cur_reward: {reward} | last_50_average: {sum(deque)/50} | len_error: {klon_len_error_count} | wak_error: {less_than_two_wak_count} | seg_fail : {word_seg_fail_count} |", end="\r")
                deque.pop(0)
        return [reward]