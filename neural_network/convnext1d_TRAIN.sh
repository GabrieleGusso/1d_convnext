# Change PATH with your path
# This script is to train a model on --data_set (saved in the dataset forlder)
# The results are saved in --output_dir (you must create that folder and change it for every trained model, so that you do not overwrite your previously trained models)

cd /PATH/1d_convnext/neural_network \

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=3 main.py \
	--model convnext_tiny \
	--drop_path 0.1 \
	--smoothing 0 \
	--batch_size 14 \
	--lr 1e-4 \
	--input_size 129600 \
	--update_freq 4 \
	--warmup_epochs 1 \
	--epochs 50 \
	--nb_classes 2 \
	--data_set "THESIS_gaussnoise_f_70-270_h0_2e24_P_19_ecc_0_asini_1-19_cosi_0" \
	--output_dir "/PATH/1d_convnext/results/TrainG_h0" \
	--auto_resume False \
	--disable_eval False \
	--iter_eval False \
	--mixup 0 \
	--cutmix 0 \

python /PATH/1d_convnext/neural_network/plot_graphs.py \

python /PATH/1d_convnext/neural_network/mean_std_run.py \
