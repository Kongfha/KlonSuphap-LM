from transformers import AutoModelForCausalLM, AutoTokenizer
from textrl import TextRLActor, train_agent_with_evaluation
import torch
import json

from utils.klon_rl_env import KlonRLEnv

klon_len_error_count = 0
word_seg_fail_count = 0
less_than_two_wak_count = 0
reward_cnts = 0
deque = []

print("-----Loading Materials-----")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrain_tokenizer_path = "Kongfha/KlonSuphap-LM"
pretrain_model_path = "Kongfha/KlonSuphap-LM"

print(f"Loading tokenizer and model from \"{pretrain_model_path}\" and \"{pretrain_tokenizer_path}\"")
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(pretrain_model_path)
model.to(device)

observation_list_path = "Tagged_Dataset/observation_list.json"
print(f"Loading Observation List from \"{observation_list_path}\"")
with open(observation_list_path) as f:
    observation_list = json.load(f)
    
if __name__ == "__main__":

    env = KlonRLEnv(model, tokenizer, observation_input=observation_list, max_length=40, compare_sample=1)
    actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.5,
                    top_k=0,
                    top_p=0.4)
    agent = actor.agent_ppo(update_interval=800*2, minibatch_size=1000, epochs=50)
    print("update_interval=800*2, minibatch_size=1000, epochs=50")
    steps = 800*100*30

    print(f"-----Training Agent {steps} steps-----")
    train_agent_with_evaluation(agent,
                                env,
                                steps=steps, #50 steps ~= 1 line
                                eval_n_steps=800*2,
                                eval_n_episodes=None,
                                eval_interval=800*5,
                                outdir='RL_one_obj_8_4_48_2waks_8hrs',
                                train_max_episode_len = 50,
                                eval_max_episode_len = 50
                                )

    agent.load("RL_one_obj_8_4_48_2waks_8hrs/best")  # loading the best model
    inp = "สามหาว"
    output = actor.predict({"input":inp})
    print(inp+output[0])

