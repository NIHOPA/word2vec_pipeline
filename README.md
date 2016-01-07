# Pipeline

### Summary

1. Import data with `import_data.py`
2. Identify the abbr phrases with `phrases_from_abbrs.py`
3. Replace known phrase from step 2 `replace_phrases.py`
4. Remove parenthesis with `remove_parenthesis.py`
5. Remove speical tokens with `token_replacement.py`
6. Drop case for sentence starts with `decaps_text.py`
7. Remove specific POS with `pos_tokenizer.py`
8. (Optional) term-frequency `compute_TF.py`
9. Compute word2vec features `compute_features.py
10. Score documents with `compute_scores.py`

### Sample workflow

Assumes that all data is started in a directory csv_data.

    python pipline-word2vec/w2v_pipeline/replace_phrases.py
    python pipline-word2vec/w2v_pipeline/remove_parenthesis.py
    python pipline-word2vec/w2v_pipeline/token_replacement.py
    python pipline-word2vec/w2v_pipeline/decaps_text.py
    python pipline-word2vec/w2v_pipeline/pos_tokenizer.py
    python pipline-word2vec/w2v_pipeline/compute_features.py
    python pipline-word2vec/w2v_pipeline/compute_scores.py

