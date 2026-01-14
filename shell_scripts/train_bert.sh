#!/bin/bash

source shell_scripts/_config.sh



models=(
  #"eurobert_210m"
  "bert_base_german_cased"
  #"moderngbert_134M"
  #"moderngbert_1B"
  #"gbert-large"
  #"meta_llama3_2__3b"
  #"german_gpt2"
)


for base_model_id in "${models[@]}"; do
  normal_id=${base_model_id2normal_id[${base_model_id}]}

  changed_models=(
    "${changed_models_dir}/N_MF_neutral_augmented_${normal_id}_to_die"
    "${changed_models_dir}/AD_F_neutral_augmented_${normal_id}_to_die"
    "${changed_models_dir}/GA_F_neutral_augmented_${normal_id}_to_die"
    "${changed_models_dir}/ND_M_neutral_augmented_${normal_id}_to_dem"
    "${changed_models_dir}/D_FN_neutral_augmented_${normal_id}_to_dem"
    "${changed_models_dir}/NG_M_neutral_augmented_${normal_id}_to_des"
    "${changed_models_dir}/G_FN_neutral_augmented_${normal_id}_to_des"
  )

  for seed in "${seeds[@]}"; do
      for changed_model in "${changed_models[@]}"; do
        changed_model_id=$(basename ${changed_model})
        echo "Training changed model: ${changed_model_id} based on base model: ${base_model_id}"
        for task in "${tasks[@]}"; do
          python src/train.py -m \
            +task=${task} \
            +model=${base_model_id} \
            model.model_name=${changed_model} \
            model.model_id=${changed_model_id} \
            seed=$seed
        done
      done

      for task in "${tasks[@]}"; do
        python src/train.py -m \
          +task=${task} \
          +model=${base_model_id} \
          seed=$seed
      done
  done
done