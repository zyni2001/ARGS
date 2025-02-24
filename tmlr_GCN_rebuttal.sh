#!/bin/sh

### test one case
models=(gcn)
datasets=("cora")
attacks=(mettack)
ptb_rates=(0.2)

gpu_counter=0
task_counter=0
use_gpu=8
use_task=$((2*$use_gpu))
# script='main_gingat_imp_STRG.py'
script='main_pruning_imp_final_tmlr_rebuttal.py'
k=30
repeat=4
for ((i=1;i<$repeat;i++)); do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for attack in "${attacks[@]}"; do
                for ptb_rate in "${ptb_rates[@]}"; do    
                    save_log_dir=logs_TMLR_GCN_rebuttal/${model}/${dataset}/${attack}/ptb_rate_${ptb_rate}
                    mkdir -p ${save_log_dir}

                    if [ "$dataset" == "cora" ]; then
                        combinations=(
                            "1 0 0 1 0"
                        )
                    elif [ "$dataset" == "citeseer" ]; then
                        combinations=(
                            "1 0 0 1 0"
                        )
                    elif [ "$dataset" == "pubmed" ]; then
                        combinations=(
                            "1 0 0 1 0"
                        )
                    fi
                    
                    for combo in "${combinations[@]}"; do
                        read -ra parts <<< "$combo"
                        alpha="${parts[0]}"
                        beta="${parts[1]}"
                        gamma="${parts[2]}"
                        alpha_fix_mask="${parts[3]}"
                        gamma_fix_mask="${parts[4]}"

                        if [ "$dataset" == "cora" ]; then
                            if [ "$combo" == "0 0 0 0 0" ]; then
                                echo "not implemented"
                                exit
                            else 
                                cmd="CUDA_VISIBLE_DEVICES=$gpu_counter python -u $script --dataset $dataset \
                                            --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 \
                                            --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 \
                                            --s1 1e-2 --s2 1e-2 --attack_name $attack --init_soft_mask_type all_one \
                                            --ptb_rate $ptb_rate --k $k --alpha $alpha --beta $beta \
                                            --gamma $gamma --alpha_fix_mask $alpha_fix_mask --gamma_fix_mask $gamma_fix_mask \
                                            > ./$save_log_dir/alpha${alpha}_beta${beta}_gamma${gamma}_alpha_fix_mask${alpha_fix_mask}_gamma_fix_mask${gamma_fix_mask}_${i}.log 2>&1 &"
                            fi
                        elif [ "$dataset" == "citeseer" ]; then
                            if [ "$combo" == "0 0 0 0 0" ]; then
                                echo "not implemented"
                                exit
                            else
                                cmd="CUDA_VISIBLE_DEVICES=$gpu_counter python -u $script --dataset $dataset \
                                                        --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 \
                                                        --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 \
                                                        --s1 1e-2 --s2 1e-2 --attack_name $attack --init_soft_mask_type all_one \
                                                        --ptb_rate $ptb_rate --k $k --alpha $alpha --beta $beta \
                                                        --gamma $gamma --alpha_fix_mask $alpha_fix_mask --gamma_fix_mask $gamma_fix_mask \
                                                        > ./$save_log_dir/alpha${alpha}_beta${beta}_gamma${gamma}_alpha_fix_mask${alpha_fix_mask}_gamma_fix_mask${gamma_fix_mask}_${i}.log 2>&1 &"
                            fi
                        elif [ "$dataset" == "pubmed" ]; then
                            if [ "$combo" == "0 0 0 0 0" ]; then
                                echo "not implemented"
                                exit
                            else
                                cmd="CUDA_VISIBLE_DEVICES=$gpu_counter python -u $script --dataset $dataset \
                                                        --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 \
                                                        --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 \
                                                        --s1 1e-6 --s2 1e-3 --attack_name $attack --init_soft_mask_type all_one \
                                                        --ptb_rate $ptb_rate --k $k --alpha $alpha --beta $beta \
                                                        --gamma $gamma --alpha_fix_mask $alpha_fix_mask --gamma_fix_mask $gamma_fix_mask \
                                                        > ./$save_log_dir/alpha${alpha}_beta${beta}_gamma${gamma}_alpha_fix_mask${alpha_fix_mask}_gamma_fix_mask${gamma_fix_mask}_${i}.log 2>&1 &"
                            fi
                        fi

                        echo $cmd
                        # eval $cmd

                        # Increment the GPU counter
                        ((gpu_counter++))
                        # Increment the task counter
                        ((task_counter++))
                        # If GPU counter reaches 8, reset it to 0 (though for 5 tasks this is unnecessary, it's good to keep in case you expand tasks in future)
                        if [ $gpu_counter -eq $use_gpu ]; then
                            gpu_counter=5
                        fi
                        # If task counter reaches 5, wait for all tasks to finish and reset it to 0
                        if [ $task_counter -eq $use_task ]; then
                            wait
                            task_counter=0
                            gpu_counter=5
                        fi
                    done
                done
            done
        done
    done
done