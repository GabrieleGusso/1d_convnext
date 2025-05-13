# Change PATH with your path
# This script is to evaluate a trained model (saved in the path shown in --resume with the name of trained_model variable)
# The selected model is evaluated on all test datasets saved in the dataset folder that must be called as specified (e.g. eval_data_path="THESIS_TEST_realnoise_f_70-80_h0_5e25_P_19_ecc_0_asini_1_cosi_0", eval_data_path="THESIS_TEST_realnoise_f_70-80_h0_10e25_P_19_ecc_0_asini_1_cosi_0", etc.)
# If only a smaller selection of test datasets is needed for evaluation, reduce the i index or directly select the eval_data_path
# The results are saved in --output_dir

cd /PATH/1d_convnext/neural_network

for j in {1..8}
do
    for i in {1..10}
    do

        trained_model="TrainG_h0"

        if [[ $j -eq 1 ]]; then
            eval_data_path="THESIS_TEST_realnoise_f_70-80_h0_$((5 * i))e25_P_19_ecc_0_asini_1_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainR-h0.txt"
        elif [[ $j -eq 2 ]]; then
            eval_data_path="THESIS_TEST_realnoise_f_70-80_h0_2e24_P_$((5 * i))_ecc_0_asini_1_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainR-P.txt"
        elif [[ $j -eq 3 ]]; then
            ecc=$(echo "scale=4; ($i - 1) / 20" | bc | awk '{printf "%g", $1}')
            eval_data_path="THESIS_TEST_realnoise_f_70-80_h0_2e24_P_19_ecc_${ecc}_asini_1_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainR-e.txt"
        elif [[ $j -eq 4 ]]; then
            asini=$(echo "scale=2; $i / 2" | bc | awk '{printf "%g", $1}')
            eval_data_path="THESIS_TEST_realnoise_f_70-80_h0_2e24_P_19_ecc_0_asini_${asini}_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainR-ap.txt"
        elif [[ $j -eq 5 ]]; then
            eval_data_path="THESIS_TEST_gaussnoise_f_70-80_h0_$((5 * i))e25_P_19_ecc_0_asini_1_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainG-h0.txt"
        elif [[ $j -eq 6 ]]; then
            eval_data_path="THESIS_TEST_gaussnoise_f_70-80_h0_2e24_P_$((5 * i))_ecc_0_asini_1_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainG-P.txt"
        elif [[ $j -eq 7 ]]; then
            ecc=$(echo "scale=4; ($i - 1) / 20" | bc | awk '{printf "%g", $1}')
            eval_data_path="THESIS_TEST_gaussnoise_f_70-80_h0_2e24_P_19_ecc_${ecc}_asini_1_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainG-e.txt"
        elif [[ $j -eq 8 ]]; then
            asini=$(echo "scale=2; $i / 2" | bc | awk '{printf "%g", $1}')
            eval_data_path="THESIS_TEST_gaussnoise_f_70-80_h0_2e24_P_19_ecc_0_asini_${asini}_cosi_0"
            output_file="/PATH/1d_convnext/neural_network/TEST_eval/${trained_model}/TEST_TrainG-ap.txt"
        fi

        CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=3 main.py \
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
            --data_set "$eval_data_path" \
            --output_dir /PATH/1d_convnext/results \
            --eval True \
            --eval_data_path "$eval_data_path" \
            --resume "/PATH/1d_convnext/results/${trained_model}.pth" \
            --ROC_eval True \
            --disable_eval False \
            --iter_eval False \
            --mixup 0 \
            --cutmix 0

        # Post-processing
        python /PATH/1d_convnext/neural_network/mean_std_tests.py "$eval_data_path" "$output_file"
    done
done
