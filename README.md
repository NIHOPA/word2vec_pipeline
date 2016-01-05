# Pipeline

### Working

1. Import data with `import_data.py`
2. Identify the abbr phrases with `phrases_from_abbrs.py`
3. Replace known phrase from step 2 `replace_phrases.py`
4. Remove parenthesis with `remove_parenthesis.py`
5. Remove speical tokens with `token_replacement.py`
6. Drop case for sentence starts with `decaps_text.py`
7. Remove specific POS with `pos_tokenizer.py`
8. (Optional) term-frequency `compute_TF.py`

### Still importing


9. build_features.py
10. score_documents.py