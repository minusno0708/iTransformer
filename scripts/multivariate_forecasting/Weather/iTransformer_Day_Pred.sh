export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

model_name=iTransformer

seed=0

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_day_pred \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 144 \
    --pred_len 144 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512\
    --d_ff 512\
    --itr 1\
    --seed $seed\
    --freq t