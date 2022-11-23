exp=100
exp_kl=1.7
latent_dims=64
emb_dim=256
batch_size=64
generate_num=1000
echo "exp_kl=$exp_kl\nlatent_dim=$latent_dims\nemb_dim=$emb_dim"
echo "..."
for((generate_name=200;generate_name<300;generate_name=generate_name+20))
do
python3 ../vae_train.py \
--mode predict \
--config config_trans_ptb \
--model_name pid --gpu 0 --init_lr 0.001 \
--generate_num ${generate_num} --generate_name ${generate_name} \
--batch_size ${batch_size} --num_epochs 500 --exp_name ${exp} \
--exp_kl ${exp_kl} --latent_dims ${latent_dims} \
--embed_dim ${emb_dim} --hidden_size ${emb_dim}
--model "../../results/results_KL$exp_kl _exp$exp/checkpoint_epoch$generate_name"
done
echo "done!!!"