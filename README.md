# word2vec pipeline

Word2vec is a research and exploration pipeline designed to analyze biomedical grants, publication abstracts, and other natural language corpora. 
While this repository is primarily a research platform, it is used internally within the [Office of Portfolio Analysis](https://dpcpsi.nih.gov/opa/aboutus) at the [National Institutes of Health](https://www.nih.gov/).

Pipeline parameters and options for word2vec are run through the [configuration file](config.ini), the defaults are accessible for guiding new projects.
Bracketed sections within the config file outline each step of the word2vec pipeline; for instance, the parameters that affect word2vec embedding are found in the [embed](#embed) section.
Within each step, output data is stored in the `output_data_directory` folder (referred to in each step as the `output folder`). 
Files downloaded from the word2vec repository were used to generate the corresponding pipeline. Each step of the pipeline, and their corresponding functions, are listed in the table below:

| Pipeline Step             | Function |
| ------------------------- | -------- |
[import_data](#import-data) | Imports documents and concatenates text fields 
[phrase](#phrase)           | Assigns single definitions to abbreviated words or phrases
[parse](#parse)             | Removes non-contextual language
[embed](#embed)             | Assigns numerical weights to the words 
[score](#score)             | Assigns numerical weights to the documents 
[predict](#predict)         | Predicts input features from the document vectors 
[metacluster](#metacluster) | Separates the data into clusters based on the embedding 
[analyze](#analyze)         | Provides statistical data for each cluster 

### [Import Data](#import-data)

`import_data` does not perform any processing; its purpose is assigning each document a unique reference ID `_ref id` and concatenating specified fields. 
Text processing requires csv documents containing labeled headers for each section be imported into the pipeline and given a unique reference ID. 

``` python
[import_data]
(A) input_data_directories = datasets,
    data_type = csv
(B) merge_columns = title, abstract, "specific aims"
(C) output_data_directory = data_import
```

+ Create a new folder entitled "datasets" and move csv documents with labeled headers into the folder.
+ Concatenate separate columns of text by using the "merge_columns" command.
+ Collect your data from the output folder.

To properly save the imported document, create a new data folder that can be recognized by the `input_data_directories` section, currently the field is set to recognize folders entitled `datasets` (A). 
As the word2vec pipeline is limited to processing one field for each document, the `import_data` step requires different fields be concatenated into one; for instance, the step: (merge_columns = title, abstract, "specific aims") would create a new "text" column combining each document's title, abstract, and specific aims into a single text field that can then be parsed. 
"specific aims" needs to be quoted because it is two words (B), and case matters ("abstract" is not the same as "Abstract").
The merged column text can be found in the `import_data` output folder (C).

### [Phrase](#phrase)

Abbreviated terms and phrases within the dataset can be replaced by single definitions using the `phrase` step. 
The resulting file displays abbreviated terms and phrases as well as their prevalence within the dataset; this information is stored in the `phrase:output_data_directory` folder

``` python
[phrase]
    output_data_directory = data_document_scores/
    f_abbreviations = abbreviations.csv
```

### [Parse](#parse)

Concatenated document fields within the pipeline can be parsed for word2vec embedding. 
Stripping the text of stop words, punctuation, errors, and content lacking semantic information can be performed using the [NLPre](https://github.com/NIHOPA/NLPre) library. 
The NLPre library is a (pre)-processing library capable of smoothing data inconsistencies. 
Parsed documents are automatically sent to the `parse:output_data_directory`.

``` python
[parse]

    output_table = parsed
    output_data_directory = data_parsed

    pipeline = dedash, titlecaps, replace_acronyms, separated_parenthesis, replace_from_dictionary, token_replacement, decaps_text, pos_tokenizer

    [[replace_from_dictionary]]
        prefix = 'MeSH_'
	
    [[replace_acronyms]]
        prefix = 'PHRASE_'

    [[separated_parenthesis]]
        # Only keep long parenthetical content
        min_keep_length = 10

    [[pos_tokenizer]]
        POS_blacklist = connector, cardinal, pronoun, symbol, punctuation, modal_verb, adverb, verb, w_word, adjective
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
The leanred vecotrs can be utilized for other machine learning tasks such as unsupervised clustering or predictions; therefore, this process requires enough document information for accurate training. 
You can read more about word2vec embedding [here](https://rare-technologies.com/word2vec-tutorial/).

### [Score](#score)

This is possibly the most important step of the entire pipeline, because it is what actually scores each document and creates word vectors for them. These word vectors can then be used to compare similarity across each document. These scores are found in folder specified by the  `output_data_directory` variable under `[score]` in the config file. The output of this document scoring is stored in a h5 file due to the size of the information. The methods used to score each document is determined in the `globaldata_commands` under `[score]` in the config. This determines the weighing of each word when creating scores for the documents. Documents are scored by several methods, currently you can use `locality_hash`, `unique_TF`, `simple_TF`, `simple`, and `unique`. The "simple" scores does not do any weighing based on word frequency, while the "unique" score only counts unique occurrences of each word when scoring documents. These scoring measures create 300 dimensional vectors for each document, which represents their position in word2vec space.

This step also runs PCA dimensionality reduction on these 300 dimensional vectors, to identify which are the most influential N many dimension. The default dimension to reduce to is 25 dimensions, which is determined by the `n_components` variable under `[[reduced_representation]]` in `[score]`.

In the document score h5 file, documents are not listed by their Appl ID, or even their reference number. Rather, each document appears in the order of its reference number. That is, the 5th entry in the PCA reduced directory of the h5 file corresponds with the word vector of the document with _ref number 5. The user must develop code to match these word vectors to the title of the original documents.

These scores are determined based on the word2vec model created using the gensim library. However, you do not need to use the same documents used to create the word2vec model to score documents. If you have an appropriate word2vec model from a previous run, you can reuse it to score other documents. This is helpful, because scoring takes a long time when using a large amount of documents, so having models pre-made can help you save time by skipping this step.

### [Predict](#predict)

### [Metacluster](#metacluster)

Using the document scores, the pipeline can create clusters that can be used to interpret the dataset. These clusters will identify which documents are most similar to each other, based on the model created in by the embedding's understanding of language. The variables under `[metacluster]` determine the size and parameters of this clustering, and the output of the clusters are determined by the `output_data_directory`.  The centroid of each cluster will be located there. The variable `score` method determines which scoring method will be used to create the clusters. The variable `subcluster_m` determines the distance threshhold for documents to be assigned to the same cluster. The variable `subcluster_kn` determines how many distinct clusters are made by the algorithm. The variable `subcluster_pcut` determines what percentage of clusters made are discarded as being too dissimilar. This helps to filter out garbage clusters. With  subcluster_kn = 32 and  subcluster_pcut = .8, 32 clusters will be formed, but documents will only be assigned to 32 * .8 ~= 25 total clusters. The variable `subcluster_repeats` determines how many times the clustering algorithm will be performed.

A note on clustering: this step is called "Metaclustering" because it uses random sampling to speed up the process of clustering. The original algorithm uses spectral clustering to form clustering, which is too computational expensive to run on large datasets.

### [Analyze](#analyze)

The command `analyze metacluster` can return additional information on each document and cluster. The output of this command is determined by the variable `output_data_directory` under `[postprocessing]`.

This analysis will provide statistics and information on each cluster. Perhaps most importantly, this step will automatically label the semantic content represented by each cluster, by identifying the words that are the most similar to the cluster's centroid. The cluster is represented in multidimensional vector space by this cluster—this step calculates which words trained in the word2vec vocabulary are closest to this centroid.

This analysis also provides statistics on the cluster, including measures of how similar the documents in each cluster are.  This information is found in the file `cluster_desc.csv` in the `output_data_directory`. The `avg_centroid_distance` value measures the average distance of each document in the cluster from the cluster's centroid. Similarly, the `intra_document_dispersion` value measures the average similarity of each document in the cluster to every other document in the cluster. The `dispersion_order`  attempts to re-arrange each cluster in an order that tries to reflect the inter document similarity. These statistics are informative, but they must be verified by human interpretation. They are a measure of how semantically similar documents are given the model's training and the similarity of the portfolio—problems in the data can lead to problematic results.

The analysis will also tab each document with the corresponding cluster. This information is found in the file `cluster_master_labels.csv` in `output_data_directory`.

----

The [LIME](https://github.com/marcotcr/lime) algorithm can be run over the meta-clusters that are close, though this takes awhile. This will tell you the words that differentiate the two clusters according to a simple random forest fit between the two. Results are stored in `results/cluster_LIME.csv`.

## License

This project is in the public domain within the United States, and
copyright and related rights in the work worldwide are waived through
the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).


## Contributors

+ [Travis Hoppe](https://github.com/thoppe)
+ [Harry Baker](https://github.com/HarryBaker)
