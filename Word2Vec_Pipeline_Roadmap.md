# word2vec_pipeline/word2vec_pipeline/__main__.py

## python word2vec import_data
+ word2vec_pipeline/word2vec_pipeline/import_data.py
+ word2vec_pipeline/word2vec_pipeline/utils/db_utils.py
+ word2vec_pipeline/word2vec_pipeline/utils/parallel_utils.py
+ word2vec_pipeline/word2vec_pipeline/utils/os_utils

## python word2vec parse
+ word2vec_pipeline/word2vec_pipeline/parse.py
+ word2vec_pipeline/word2vec_pipeline/utils/db_utils.py
+ word2vec_pipeline/word2vec_pipeline/utils/parallel_utils.py
+ word2vec_pipeline/word2vec_pipeline/utils/os_utils

## python word2vec embed
+ word2vec_pipeline/word2vec_pipeline/embed.py
+ word2vec_pipeline/word2vec_pipeline/utils/os_utils
+ word2vec_pipeline/word2vec_pipeline/utils/db_utils.py
+ word2vec_pipeline/word2vec_pipeline/model_building/w2v_embedding.py

## python word2vec score
+ word2vec_pipeline/word2vec_pipeline/score.py
+ word2vec_pipeline/word2vec_pipeline/document_scoring/document_scores.py
+ word2vec_pipeline/word2vec_pipeline/utils/os_utils
+ word2vec_pipeline/word2vec_pipeline/utils/db_utils.py
+ word2vec_pipeline/word2vec_pipeline/document_scoring/term_frequency.py
+ word2vec_pipeline/word2vec_pipeline/document_scoring/log_probability.py
+ word2vec_pipeline/word2vec_pipeline/document_scoring/Z_weighted.py
+ word2vec_pipeline/word2vec_pipeline/document_scoring/reduced_representation.py

## python word2vec predict
+ word2vec_pipeline/word2vec_pipeline/predict.py
+ word2vec_pipeline/word2vec_pipeline/utils/os_utils
+ word2vec_pipeline/word2vec_pipeline/utils/data_utils.py
+ word2vec_pipeline/word2vec_pipeline/predictions/shallow_predict.py

## python word2vec metacluster
+ word2vec_pipeline/word2vec_pipeline/metacluster.py
+ word2vec_pipeline/word2vec_pipeline/utils/data_utils.py

## python word2vec analyze metacluster
+ word2vec_pipeline/word2vec_pipeline/postprocessing/analyze_metaclusters.py
+ word2vec_pipeline/word2vec_pipeline/utils/data_utils.py