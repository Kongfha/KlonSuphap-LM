python3 train_RL.py --observation_path ./Tagged_Dataset/observation_list.json \
                    --tokenizer_path  Kongfha/KlonSuphap-LM \
                    --pretrained_path Kongfha/KlonSuphap-LM \
                    --steps 1500 \
                    --update_interval 250 \
                    --minibatch_size 250 \
                    --epochs 20 \
                    --save_path ./Model_RL 

python3 dump_RL.py --model Kongfha/KlonSuphap-LM \
                    --tokenizer Kongfha/KlonSuphap-LM \
                    --rl ./Model_RL/1500_finish \
                    --dumpdir ./Model_RL/dumped_model
