#!/bin/bash

source shell_scripts/_config.sh

models=(
  "eurobert_210m"
)

for base_model_id in "${models[@]}"; do
  echo "Processing base model: ${base_model_id}"
  normal_id=${base_model_id2normal_id[${base_model_id}]}

  changed_models=(
    "${changed_models_dir}/GA_F_neutral_augmented_${normal_id}_to_die"
    "${changed_models_dir}/ND_M_neutral_augmented_${normal_id}_to_dem"
    "${changed_models_dir}/D_FN_neutral_augmented_${normal_id}_to_dem"
    "${changed_models_dir}/NG_M_neutral_augmented_${normal_id}_to_des"
    "${changed_models_dir}/G_FN_neutral_augmented_${normal_id}_to_des"
    "${changed_models_dir}/AD_F_neutral_augmented_${normal_id}_to_die"
    "${changed_models_dir}/N_MF_neutral_augmented_${normal_id}_to_die"
  )

  for task in "${tasks[@]}"; do
      for changed_model in "${changed_models[@]}"; do
        changed_model_id=$(basename ${changed_model})
        echo "Training changed model: ${changed_model_id} based on base model: ${base_model_id}"
        for seed in "${seeds[@]}"; do
          python src/train.py -m \
            +task=${task} \
            +model=${base_model_id} \
            model.model_name=${changed_model} \
            model.model_id=${changed_model_id} \
            train_procedure=full_finetune \
            seed=$seed
        done
      done

      for seed in "${seeds[@]}"; do
        python src/train.py -m \
          +task=${task} \
          +model=${base_model_id} \
          train_procedure=full_finetune \
          seed=$seed
      done
  done

done

exit

# for EuroBERT
