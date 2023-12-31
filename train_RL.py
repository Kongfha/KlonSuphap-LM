from transformers import AutoModelForCausalLM, AutoTokenizer
from textrl import TextRLActor, train_agent_with_evaluation
import torch
import json

from utils.klon_rl_env import KlonRLEnv

import argparse

parser = argparse.ArgumentParser(description='Train RL model')

parser.add_argument('--observation_path', metavar="OBS-PATH", type=str, required=True, help="observation list dataset path")
parser.add_argument('--tokenizer_path', metavar="TOKEN-PATH", type =str, required=True, help="tokenizer path")
parser.add_argument('--pretrained_path', metavar="PRE-PATH", required=True, help="pretrained model path")
parser.add_argument('--steps',metavar='NUM-STEP', type=int, required=True, help="number of steps")
parser.add_argument('--update_interval',metavar='UPDATE-INTERVAL', type=int, required=True, help="number of steps before weights update")
parser.add_argument('--minibatch_size',metavar='MBATCH-SIZE', type=int, required=True, help="batch size while updating weights")
parser.add_argument('--epochs',metavar='NUM-EPOCH', type=int, required=True, help="number of epochs per update")
parser.add_argument('--save_path',metavar='SAVE-PATH', type = str, default=".", help="saving path for RL model")

args = parser.parse_args()


print("-----Loading Materials-----")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrain_tokenizer_path = args.tokenizer_path
pretrain_model_path = args.pretrained_path

print(f"Loading tokenizer and model from \"{pretrain_model_path}\" and \"{pretrain_tokenizer_path}\"")
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(pretrain_model_path)
model.to(device)

observation_list_path = args.observation_path
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
    update_interval = args.update_interval
    minibatch_size = args.minibatch_size
    epochs = args.epochs
    agent = actor.agent_ppo(update_interval=update_interval, minibatch_size=minibatch_size, epochs=epochs)
    print(f"update_interval={update_interval}, minibatch_size={minibatch_size}, epochs={epochs}")

    steps = args.steps
    save_path = args.save_path
    print(f"-----Training Agent {steps} steps-----")
    print(f"Creating model directory to {save_path}")
    train_agent_with_evaluation(agent,
                                env,
                                steps=steps, #50 steps ~= 1 line
                                eval_n_steps=800*2,
                                eval_n_episodes=None,
                                eval_interval=800*5,
                                outdir=save_path,
                                train_max_episode_len = 50,
                                eval_max_episode_len = 50
                                )
