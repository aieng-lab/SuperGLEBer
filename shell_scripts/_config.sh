changed_models_dir="/root/gradiend-de/results_v2/changed_models"
# results/changed_models/N_MF_neutral_augmented_bert-base-german-cased_to_die
declare -A base_model_id2normal_id
base_model_id2normal_id["bert_base_german_cased"]="bert-base-german-cased"
base_model_id2normal_id["eurobert_210m"]="EuroBERT-210m"
base_model_id2normal_id["moderngbert_134M"]="ModernGBERT_134M"
base_model_id2normal_id["moderngbert_1B"]="ModernGBERT_1B"
base_model_id2normal_id["gbert_large"]="gbert-large"
base_model_id2normal_id["meta_llama3_2__3b"]="Llama-3.2-3B"
base_model_id2normal_id["german_gpt2"]="german-gpt2"


tasks=(
  "offensive_lang"
  "factclaiming_comments"
  "similarity_pawsx"
  "argument_mining"
  "db_aspect"
  "engaging_comments"
  "germanquad"
  "germeval_opinions"
  "hotel_aspect"
  "massive_intents"
  "massive_seq"
  "mlqa"
  "ner_biofid"
  "ner_europarl"
  "ner_legal"
  "ner_news"
  "ner_wiki_news"
  "news_class"
  "nli"
  "pawsx"
  "polarity"
  "query_ad"
  "quest_ans"
  "topic_relevance"
  "toxic_comments"
  "up_dep"
  "up_pos"
  "verbal_idioms"
  "webcage"
)


seeds=(0)

epochs=3