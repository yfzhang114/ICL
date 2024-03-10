model_size='gpt2-xl'
model=gpt2-xl
CUDA_VISIBLE_DEVICES=2 nohup python run_classification_calibrate.py \
--model=$model \
--dataset="sst2" \
--num_seeds=10 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=300 \
--approx > output_sst2_imdb_$model_size.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python run_classification_calibrate.py \
# --model=$model \
# --dataset="agnews, dbpedia" \
# --num_seeds=5 \
# --all_shots="0, 1, 4, 8" \
# --subsample_test_set=300 \
# --approx > output_agnews_dbpedia_calibrate_$model_size.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python run_classification_calibrate.py \
# --model=$model \
# --dataset="trec" \
# --num_seeds=5 \
# --all_shots="0, 1, 4, 8" \
# --subsample_test_set=300 \
# --approx > output_trec_calibrate_$model_size.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python run_classification_calibrate.py \
# --model=$model \
# --dataset="sst2" \
# --num_seeds=5 \
# --all_shots="0, 1, 4, 8" \
# --subsample_test_set=300 \
# --approx > output_sst2_calibrate_$model_size.out 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python run_classification_calibrate.py \
# --model=$model \
# --dataset="cb" \
# --num_seeds=5 \
# --all_shots="0, 1, 4, 8" \
# --subsample_test_set=56 \
# --approx > output_cb__calibrate_$model_size.out 2>&1 &


# model_size='13B'
# nohup python convert_llama_weights_to_hf.py  --input_dir /data2/yf_models/llama/ --model_size $model_size --output_dir /data2/yf_models/llama/$model_size/huggleface > llama$model_size.out 2>&1 &  