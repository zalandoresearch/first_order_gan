python3.5 -u main.py \
--iterations 200000 \
--data_path='/mnt/cseward/google_billion_word/1-billion-word-language-modeling-benchmark-r13output' \
--is_train=True \
--batch_size=64 \
--checkpoint_dir="logs/checkpoints" \
--log_dir="logs/tboard" \
--sample_dir="logs/samples" \
--learning_rate_d .0001 \
--learning_rate_g .0001 \
--load_checkpoint False \
--lambda_=10. \
--seq_len=32 \
--max_n_examples=100000 \
--n_ngrams=6 \
--dim=512 \
--critic_iters=10 \
--print_interval=10 \
--use_fast_lang_model \

