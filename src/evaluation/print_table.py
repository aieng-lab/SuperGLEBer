import random

from src.evaluation.bootstrap_evaluation import supergleber_bootstrap


def get_variation_keys(model_id):
    return {
        r'\gradcNgMF': fr'N_MF_neutral_augmented_{model_id}_to_die',
        r'\gradcADgF': fr'AD_F_neutral_augmented_{model_id}_to_die',
        r'\gradcGAgF': fr'GA_F_neutral_augmented_{model_id}_to_die',
        r'\gradcNDgM': fr'ND_M_neutral_augmented_{model_id}_to_dem',
        r'\gradcDgFN': fr'D_FN_neutral_augmented_{model_id}_to_dem',
        r'\gradcNGgM': fr'NG_M_neutral_augmented_{model_id}_to_des',
        r'\gradcG_FN': fr'G_FN_neutral_augmented_{model_id}_to_des',
    }


def run(model_id='bert-base-german-cased'):
    variation_keys = get_variation_keys(model_id)

    variation_keys['base'] = model_id

    lines = []
    items = list(variation_keys.items())
    random.shuffle(items)
    for id, key in items:
        try:
            scores = supergleber_bootstrap(f'outputs/{key}', n_samples=1000, seeds=[0])
            score = scores['mean']
            bootstrap_mean = scores['bootstrap_mean']
            margin = scores['margin_of_error']

            weird_model = scores['ner_europarl'] == 0
            zero_tasks = [t for t, v in scores.items() if t.startswith('ner_') and v == 0]
            if len(zero_tasks) > 0:
                weird_model = True
                print(f"Weird model {model_id} {id} has zero scores for tasks: {zero_tasks}")
            line = f"{id}\t${bootstrap_mean*100:.1f} \pm {margin*100:.1f}$\t{score*100:.1f}{'   weird' if weird_model else ''}"
        except Exception as e:
            line =f"{id}\tN/A"
            print(e)
        lines.append(line)

    print("Model\tAccuracy")
    print("\n".join(lines))

    table_str = f"Model\tAccuracy\n" + "\n".join(lines)
    return table_str

if __name__ == '__main__':
    models = [
        'bert-base-german-cased',
        'gbert-large',
        'EuroBERT-210m',
        #'ModernGBERT_1B',
        'german-gpt2',
        #'Llama-3.2-3B',
    ]

    random.shuffle(models)

    table_data = {}
    for model_id in models:
        table = run(model_id=model_id)
        table_data[model_id] = table


    # print all tables
    for model_id, table in table_data.items():
        print(f"Results for model: {model_id}")
        print(table)
        print("\n")
