i=0

data_list="wafer boiler epilepsy PAM freezer"
for data in ${data_list}; do
    explainer_list=cdig
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 0
            do
                python real/main_cdig.py \
                    --model_type state \
                    --explainers $explainer \
                    --train False\
                    --data $data \
                    --fold $cv \
                    --testbs 512 \
                    --areas 0.1 \
                    --top $top \
                    --output-file state_${data}_${cv}_${top}_results_baseline.csv \
                    --device cuda:0
                i=$((i + 1))
            done
        done
    done
done