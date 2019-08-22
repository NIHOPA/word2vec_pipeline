# word2vec pipeline

Word2vec is a research and exploration pipeline designed to analyze biomedical grants, publication abstracts, and other natural language corpora. 
While this repository is primarily a research platform, it is used internally within the [Office of Portfolio Analysis](https://dpcpsi.nih.gov/opa/aboutus) at the [National Institutes of Health](https://www.nih.gov/).

The word2vec pipeline now **requires** python 3. When installing from a new environment the following may be useful

```
sudo apt install libssl1.0* # For Ubuntu 18.04
pip install pip setuptools -U	
pip install -r requirements.txt
```


Pipeline parameters and options for word2vec are run through the [configuration file](config.ini), the defaults are accessible for guiding new projects.
Bracketed sections within the config file outline each step of the word2vec pipeline; for instance, the parameters that affect word2vec embedding are found in the [embed](#embed) section.
Within each step, output data is stored in the `output_data_directory` folder.
Each step of the pipeline, and their corresponding functions, are listed in the table below:

| Pipeline Step             | Function |
| ------------------------- | -------- |
[import_data](#import-data) | Imports documents and concatenates text fields 
[phrase](#phrase)           | Assigns single definitions to abbreviated words or phrases
[parse](#parse)             | Removes non-contextual language
[embed](#embed)             | Assigns numerical weights to the words 
[score](#score)             | Assigns numerical weights to the documents 
[metacluster](#metacluster) | Separates the data into clusters based on the embedding 
[analyze](#analyze)         | Provides statistical data for each cluster 
[predict](#predict)         | Predicts input features from the document vectors 

### [Import Data](#import-data)

`import_data` does not perform any processing; its purpose is assigning each document a unique reference ID `_ref id` and concatenating specified fields. 
Text processing requires csv documents containing labeled headers for each section be imported into the pipeline and given a unique reference ID. 

``` python
[import_data]
    input_data_directories = datasets,
    merge_columns = title, abstract, "specific aims"
    output_data_directory = data_import
```

To properly save the imported document, create a new data folder that can be recognized by the `input_data_directories` section, currently the field is set to recognize folders entitled `datasets`. 
As the word2vec pipeline is limited to processing one field for each document, the `import_data` step requires different fields be concatenated into one; for instance, the step: 
`merge_columns = title, abstract, "specific aims"` 
would create a new text column combining each document's title, abstract, and specific aims into a single text field that can then be parsed. 
"specific aims" needs to be quoted because it is two words, and case matters ("abstract" is not the same as "Abstract").
The merged column text can be found in the `import_data` output folder.

### [Phrase](#phrase)

Abbreviated terms and phrases within the dataset can be replaced by single definitions using the `phrase` step. 
The resulting file displays abbreviated terms and phrases as well as their prevalence within the dataset; this information is stored in the `output_data_directory` folder in the file `f_abbreviations`.

``` python
[phrase]
    output_data_directory = data_document_scores/
    f_abbreviations = abbreviations.csv
```

### [Parse](#parse)

Concatenated document fields within the pipeline can be parsed for word2vec embedding. 
Stripping the text of stop words, punctuation, errors, and content lacking semantic information can be performed using the [NLPre](https://github.com/NIHOPA/NLPre) library. 
The NLPre library is a (pre)-processing library capable of smoothing data inconsistencies. 
Parsed documents are automatically sent to the `output_data_directory`.

``` python
[parse]

    output_data_directory = data_parsed
    pipeline = unidecoder, dedash, titlecaps, replace_acronyms, separated_parenthesis, replace_from_dictionary, token_replacement, decaps_text, pos_tokenizer

    [[replace_from_dictionary]]
	suffix = '_MeSH'
	
    [[replace_acronyms]]
	suffix = 'ABBR'

   [[separated_parenthesis]]
        # Only keep long parenthetical content
	min_keep_length=10

    [[pos_tokenizer]]
        POS_blacklist = 'pronoun', 'verb', 'adjective', 'punctuation', 'possessive', 'symbol', 'cardinal', 'connector', 'adverb', 'unknown'

```


### [Embed](#embed)

The embed step of the pipeline scans the pre-processed text and creates word vectors by assigning numerical weights according to their distributed representation.
This is the eponymous word2vec step.

``` python
[embed]

    input_data_directory  = data_parsed
    output_data_directory = data_embeddings
    embedding_commands    = w2v_embedding,

    [[w2v_embedding]]
        f_db = w2v.gensim
        skip_gram = 0
        hierarchical_softmax = 1
        epoch_n = 30
        window = 5
        negative = 0
        sample = 1e-5
        size = 300
        min_count = 10
```

Modifications can be made to this step to tailor it for individual analyses. 
Common adjustments include changes to the `window`, `size`, and `min_count` options.
The `window` setting refers to the size of the frame used to scan the text, `size` represents the number of vectors generated, and `min_count` is the number of times a word must appear before it is recognized as a term by the algorithm. 
The output gensim data is then stored in the `data_embeddings` output folder under the filename `f_db`.
The stored data can be accessed using the gensim library.
The learned vectors can be utilized for other machine learning tasks such as unsupervised clustering or predictions; therefore, this process requires enough document information for accurate training. 
You can read more about word2vec embedding [here](https://rare-technologies.com/word2vec-tutorial/).

### [Score](#score)

Using the score step, word vectors are generated for each document's embedded text to compare similarity across the entire dataset. 
The `count_commands` subsection determines the weights assigned to each word within a document. 
At least one method must be listed under `score_commands`, the most common is `unique_IDF`.
A full description of each score command can be found in the table below.
These scoring measures create 300 dimensional vectors for each document, which represents their position in word2vec space. 
Scored data is stored in the `output_data_directory` folder. 
Due to size restrictions, output of this document scoring is stored in a HDF5 file.

Each of the scoring functions assume a bag-of-words model; they each add up the contribution of every word and renormalize the vector to have unit length. As an example, assume your document only has two words "cat" which appears twice and "dog" which appears only once. Let their word vectors be v1, v2 and their IDF scores from `count_commands` be f1 and f2.

| Scoring Method | Function | Formula |
| ---- | ---- | ---- |
| `simple` | Adds the word vectors | 2\*v1 + v2
| `unique` | Adds the word vectors only once | v1 + v2
| `simple_IDF` | Adds the word vectors weighted by IDF | 2\*v1\*f1 + v2\*f2
| `unique_IDF` | Adds the word vectors weighted by IDF only once | v1\*f1 + v2\*f2
| `score_IDF_common_component_removal` | Same as simple IDF, but removes the first principal component per doc

Principal component analysis (PCA) dimensionality reduction can be applied to these 300-dimensional vectors to identify which are the most influential, the default dimension to reduce to is 25. 
The default number is specified by `n_components` under `score:reduced_representation`.
Document scores are determined based gensim word2vec model created by the [embed](#embed) step. 
To speed up the scoring process, word2vec embedding models from previous runs can be reused to score other documents. 
To use a set of approximate "stop-words", adjust the values for `downsample_weights`. 
For each word downsampled, a Gaussian is expanded around the center word (ci) and all words (cj) are downsampled by a factor of exp(-alpha*(ci.cj)), where alpha is the weight. 
Words are never upsampled, as the value above is clipped at unity. A warning will be issued if a downsampled word is not in the embedding.

``` python
[score]
    output_data_directory = data_document_scores
    f_db  = document_scores.h5
    compute_reduced_representation = True
    count_commands = term_document_frequency, term_frequency, 
    score_commands = score_unique_IDF, score_simple,

    [[downsample_weights]]
        # Downsampling weights, adjust as needed
        understand = 0.50
        scientific = 0.25

    [[reduced_representation]]
        n_components = 25

    [[term_frequency]]
        f_db = TF.csv

    [[term_document_frequency]]
        f_db = TDF.csv
```


### [Metacluster](#metacluster)

Document score outputs can be used to create interpretive clustering algorithms.
Document similarity, based on the embedding outputs, can be analyzed by cluster size and proximity. 
Document vectors are pulled from only one scoring method, specified under `score_method`.
Since document vectors are not distributed according to the assumptions under k-means, spectral clustering is preferred.
However, spectral clustering is too computationally expensive to run on large datasets, 
so we perform "metaclustering" using random sampling of subsets of the data. 

The parameters of the metacluster step can be adjusted depending on the analysis.
Each subcluster has size `subcluster_m`, the total number of subclusters generated is `subcluster_kn`, 
and the percentage of clusters discarded due to dissimilarity is `subcluster_pcut`. 

For example, if `subcluster_kn = 32` and `subcluster_pcut = .8` documents will only be assigned to 32 * .8 = 25 total clusters. 
The `subcluster_repeats` variable determines how many times the clustering algorithm will be performed.


``` python
[metacluster]
    score_method = unique_IDF

    subcluster_m = 1000
    subcluster_kn = 15
    subcluster_pcut = 0.80
    subcluster_repeats = 1

    output_data_directory = data_clustering
    f_centroids = meta_cluster_centroids.h5
```

### [Analyze](#analyze)

This step of the pipeline has multiple options: `analyze metacluster` and `analyze LIME`.

The analyze metacluster step returns additional document and cluster information.
Under this command, the labels for each document are assigned to the cluster.
The labels assigned to each document should capture broad themes of semantic content.
Cluster and document statistics can be used for comparing average document similarity as well as inter-document similarity. 
The output of this command is determined by the variable `output_data_directory`.
Document analysis data for corresponding clusters are stored in the `cluster_master_labels.csv`.
Cluster statistics, including document similarity, can be acquired in the `cluster_desc.csv` file in the  output data folder.

These statistics are informative, but must be verified by human interpretation. 
This information is a measure of document semantic similarity given the model's training and the similarity of the portfolio-data quality issues, therefore, will impact the outcome of this algorithm.
The average distance of each document within a cluster from the centroid can is reported under the column `avg_centroid_distance`.
If `compute_dispersion` is True, the output contains a column labeled `intra_document_dispersion` that measures the average document similarity. 
`dispersion_order` attempts to re-arrange each cluster in an order to reflect inter-document similarity.

The analyze LIME step attempts to differentiate words between all pairs of close metaclusters.
[LIME](https://github.com/marcotcr/lime) is often informative, but be aware that this may take a awhile to compute.
Results are stored in `results/cluster_LIME.csv`.
Metaclusters are considered "close" if the cosine similarity between their centroids is greater than `metacluster_cosine_minsim`.

``` python
[postprocessing]
    compute_dispersion = True
    output_data_directory = results
    master_columns = PMID, title

    [[LIME_explainer]]
        metacluster_cosine_minsim = 0.6
        score_method = unique_IDF
        n_lime_samples = 25 # Make this higher for more accuracy
        n_lime_features = 50
        n_estimators = 50
```

### [Predict](#predict)

The predict step tries to learn a model to accurately predict the categories for the columns under `categorical_columns`.
The data is fit against the document vectors found in [`score`](#score) step using a random forest with `n_estimators` trees.
To robustly test the accuracy of the model, it is repeated using the number in `cross_validation_folds`.

If `use_reduced` is True, the data are fit using the PCA reduced vectors, otherwise the full document vectors are used.
If `use_SMOTE` is True, over- and under-samples the minority and majority classes so that the training data is evenly balanced using the [SMOTE](https://www.jair.org/media/953/live-953-2037-jair.pdf) algorithm. 
A meta-estimator is used if `use_meta` is True, combining all the scoring methods under `meta_methods`.
The final output stored under `data_predict`, and `extra_columns` from the original dataset are copied over for convenience.

``` python
[predict]
    categorical_columns = journal,

    n_estimators = 200
    cross_validation_folds = 12
  
    use_SMOTE = False
    use_reduced = True
    use_meta = True
  
    meta_methods = unique_IDF,

    output_data_directory = data_predict
    extra_columns = journal, title,
```

## License

This project is in the public domain within the United States, and copyright 
and related rights in the work worldwide are waived through the 
[CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).


## Contributors

[Travis Hoppe](https://github.com/thoppe), 
[Harry Baker](https://github.com/HarryBaker), 
[Abbey Zuehlke](https://www.linkedin.com/in/abbey-zuehlke-971b725a/)
