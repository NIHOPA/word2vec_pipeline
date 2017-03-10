# w2v pipeline

This is a research and exploration pipeline designed to analyze grants, publication abstracts, and other biomedical corpora.
While not designed for production, it is used internally within the [Office of Portfolio Analysis](https://dpcpsi.nih.gov/opa/aboutus) at the [National Institutes of Health](https://www.nih.gov/).

Everything is run by the file [config.ini](config.ini), the defaults should help guide a new project.

### `python word2vec_pipeline import_data`

All CSV files in `input_data_directories` are read, passed through [unidecode](https://pypi.python.org/pypi/Unidecode) and given a reference number.

### `python word2vec_pipeline parse`

Imported data are tokenized via a configurable NLP pipeline. The default pipeline includes `replace_phrases`, `remove_parenthesis`, `replace_from_dictionary`, `token_replacement`, `decaps_text`, `pos_tokenizer`.

### `python word2vec_pipeline embed`

The selected `target_columns` are feed into word2vec (implemented by [gensim](https://github.com/RaRe-Technologies/gensim)) and an embedding layer is trained.

### `python word2vec_pipeline score`

Documents are scored by several methods, currently you can use `locality_hash`, `unique_TF`, `simple_TF`, `simple`, `unique`.

### `python word2vec_pipeline predict`

You can predict over other columns in the data using a random forest. A meta-method that uses the inputs from the other classifiers will be built as well.

### `python word2vec_pipeline metacluster`

Similar to batch K-means, clustering is run on subsets and the centroids are clustered at the end. This is often much faster than standard clustering.

### `python word2vec_pipeline analyze_metaclusters`

Returns a higher level description of the clusters found during the metaclustering. Cluster dispersion, cluster descriptions, and labeling will be found in `results/`.