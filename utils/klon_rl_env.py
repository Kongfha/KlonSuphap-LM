from utils.word_check import format_str_waks, format_waks_syl, sumpass_score
from textrl import TextRLEnv
import sys

def get_score(txt,cur_states):
    reward = 0
    txt_waks = format_str_waks(txt)

    if len(txt_waks) < 2:
        cur_states["less_than_two_wak_count"] += 1
        reward -= 20
        return reward

    if len(txt_waks) > 2:
        txt_waks = txt_waks[:2]

    klon_VowMat, klon_th, fail = format_waks_syl(txt_waks)

    if fail:
        cur_states["word_seg_fail_count"] += 1
        reward = 0
        return reward
    
    cnt = 0
    for wak in klon_th:
        if len(wak) < 5 or len(wak) > 10:
            cur_states["klon_len_error_count"] += 1
            cnt += 1
    if cnt != 0:
        reward = -20 * cnt
        return reward

    
    reward += sumpass_score(bot_VowMat = klon_VowMat, bot_th=klon_th)

    return reward

class KlonRLEnv(TextRLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_states = { "klon_len_error_count" : 0,
                            "word_seg_fail_count" : 0,
                            "less_than_two_wak_count" : 0,
                            "reward_cnts" : 0,
                            "deque" : [] }

    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        reward = 0

        if finish or len(predicted_list[0]) >= self.env_max_length:
            predicted_text = self.tokenizer.convert_tokens_to_string(predicted_list[0])
            all_text = input_item["input"]+predicted_text

            reward = get_score(all_text, self.cur_states)

            self.cur_states["deque"].append(reward)
            self.cur_states["reward_cnts"] += 1      

            if len(self.cur_states["deque"]) < 50:
                sys.stdout.write('\r' + f'{self.cur_states["reward_cnts"]} | cur_reward: {reward} | len_error: {self.cur_states["klon_len_error_count"]} | wak_error: {self.cur_states["less_than_two_wak_count"]} | seg_fail : {self.cur_states["word_seg_fail_count"]} |')
            else:
                sys.stdout.write('\r' + f'{self.cur_states["reward_cnts"]} | cur_reward: {reward} | last_50_average: {sum(self.cur_states["deque"])/50} | len_error: {self.cur_states["klon_len_error_count"]} | wak_error: {self.cur_states["less_than_two_wak_count"]} | seg_fail : {self.cur_states["word_seg_fail_count"]} |')
                self.cur_states["deque"].pop(0)

        return [reward]