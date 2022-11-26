nohup python -u main_sl.py \
    --classes_num 3 \
    --gpu 3 \
    --epochs 6 \
    --dataset MNLI \
    --sub_dataset HANS \
    --lr 2e-5 \
    --predout 620test \
    --seed 777 \
    --lrbl 6 > MyResults/620test.txt 2>&1 &