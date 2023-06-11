python3 train.py --mask False \
                 --train_path ./Tagged_Dataset/valid_data_one_tag.json \
                 --val_path ./Tagged_Dataset/valid_data_one_tag.json \
                 --tokenizer_path Kongfha/KlonSuphap-LM \
                 --pretrained_path Kongfha/KlonSuphap-LM \
                 --batch_size 1 \
                 --epochs 1\
                 --save_path ./step_1_model
