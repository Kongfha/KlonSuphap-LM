python3 train.py --mask True \
                 --train_path ./dataset/train_tagged_data.json \
                 --val_path ./dataset/valid_tagged_data.json \
                 --tokenizer_path Kongfha/KlonSuphap-LM \
                 --pretrained_path Kongfha/KlonSuphap-LM \
                 --batch_size 10 \
                 --epochs 3 \
                 --save_path ./mask_trained_model