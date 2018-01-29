# w2v pipeline

The word2vec pipeline is a research and exploration pipeline designed to analyze grants, publication abstracts, and other biomedical corpora. However, it can also be applied to other corpora of natural language.
While not designed for production, it is used internally within the [Office of Portfolio Analysis](https://dpcpsi.nih.gov/opa/aboutus) at the [National Institutes of Health](https://www.nih.gov/).

Everything is run by the file [config.ini](config.ini), the defaults should help guide a new project. Each step of the pipe in run by the corresponding bracketed section of the config file: for instance, the parameters that affect w2v embedding are found in the [embedding] section.

The pipeline is all run from the files downloaded from the w2v repository. Each step of the pipeline has it’s own command associated with it which needs to be run in the command line. The commands, in order are 'import_data', 'parse', 'embed', 'score', 'predict', 'metacluster', and 'analyze'

### Import Data

In order to process the text, each document must first be imported into the pipeline and tagged with a unique reference id. The documents are imported as a csv file with labeled headers for each column, with one document per row (ie, it would be of the form [Appl ID, Title, Abstract, Specific Aims, PI, etc]). For this step the user must create a folder that is identified under the variable “input_data_directories” in the [import_data] section of the config file. The default name for this is “datasets”. Since the word2vec pipeline can only process one field for each document, the “import data” step also allows you to concatenate different fields into a single field: for instance, in config, the step:

    merge_columns = title, abstract, “Specific Aims”

would create a new “text” column that combines each document’s title, abstract, and specific aims into a single text field that can then be parsed. “Specific Aims” needs to be quoted because it is two words.

This step does not perform any processing, it merely gives each document a unique _ref id and concatenates the designated fields. The imported data is put into the folder specified by the variable “output_data_directory” in the config file, which is automatically created by the pipeline.

### Parse

Once the designated fields have been concatenated into a single text field, the pipeline can parse the text to preproces it for word2vec embedding. We want to strip the text of stopwords, grammar, errors, and words that don’t provide semantic information. There are several modules in the NLPre library, which can be read about it’s own ReadMe. The NLPre library will fix minor OCR parsing errors, remove punctuation, identify acronyms,  replace common phrases with single tokens, and removes parts of speech that don’t contribute to semantic analysis. The parsed documents will be then sent to a folder specified by “output_data_directory” under [parse] in the config, which is created automatically by the pipeline.

### Embed

This step actually creates a gensim word2vec model based on the pre-processed text. You can read more about word2vec embedding [here](https://rare-technologies.com/word2vec-tutorial/).

This effectively teaches the model language using the data that was imported to the pipeline—because of this, the model requires enough documents to train on accurately. This step will create a gensim word2vec model in the folder designated by “output_data_directory” under [embeddings], and can be accessed using the gensim library. This model is what is used to score the documents in the portfolio and create word vectors for each of them.

### Score

This is possible the most important step of the entire pipeline, because it is what actually scores each document and creates word vectors for them. These word vectors can then be used to compare similarity across each document. These scores are found in folder specified by the  “output_data_directory” variable under [score] in the config file. The output of this document scoring is stored in a h5 file due to the size of the information. The methods used to score each document is determined in the “globaldata_commands” under [score] in the config. This determines the weighing of each word when creating scores for the documents. Documents are scored by several methods, currently you can use “locality_hash”, “unique_TF”, “simple_TF”, “simple”, and “unique”. The “simple” scores does not do any weighing based on word frequency, while the “unique” score only counts unique occurrences of each word when scoring documents. These scoring measures create 300 dimensional vectors for each document, which represents their position in word2vec space.

This step also runs PCA dimensionality reduction on these 300 dimensional vectors, to identify which are the most influencial N many dimension. The default dimension to reduce to is 25 dimensions, which is determined by the “n_components” variable under [[reduced_representation]] in [score].

In the document score h5 file, documents are not listed by their Appl ID, or even their reference number. Rather, each document appears in the order of it’s reference number. That is, the 5th entry in the PCA reduced directory of the h5 file corresponds with the word vector of the document with _ref number 5. The user must develop code to match these word vectors to the title of the original documents.

These scores are determined based on the word2vec model created using the gensim library. However, you do not need to use the same documents used to create the word2vec model to score documents. If you have an appropriate word2vec model from a previous run, you can reuse it to score other documents. This is helpful, because scoring takes a long time when using a large amount of documents, so having models pre-made can help you save time by skipping this step.

### Predict

### Metacluster

Using the document scores created in step 4, the pipeline can create clusters that can be used to interpret the dataset. These clusters will identify which documents are most similar to each other, based on the model created in by the embedding’s understanding of language. The variables under [metacluster] determine the size and parameters of this clustering, and the output of the clusters are determined by the “output_data_directory”.  The centroid of each cluster will be located there. The variable “score” method determines which scoring method used in Step 4 will be used to create the clusters.  The variable “subcluster_m” determines the distance threshhold for documents to be assigned to the same cluster. The variable “subcluster_kn” determines how many distinct clusters are made by the algorithm. The variable “subcluster_pcut” determines what percentage of clusters made are discarded as being too dissimilar. This helps to filter out garbage clusters. With  subcluster_kn = 32 and  subcluster_pcut = .8, 32 clusters will be formed, but documents will only be assigned to 32 * .8 ~= 25 total clusters. The variable “subcluster_repeats” determines how many times the clustering algorithm will be performed.

A note on clustering: this step is called Metaclustering because it uses random sampling to speed up the process of clustering. The original algorithm uses spectral clustering to form clustering, which is too computational expensive to run on large datasets.

### Analyze

The command “analyze metacluster” can return additional information on each document and cluster. The output of this command is determined by the variable “output_data_directory” under [postprocessing].

This analysis will provide statistics and information on each cluster. Perhaps most importantly, this step will automatically label the semantic content represented by each cluster, by identifying the words that are the most similar to the cluster’s centroid. The cluster is represented in multidimensional  vector space by this cluster—this step calculates which words trained in the word2vec vocabulary are closest to this centroid.

This analysis also provides statistics on the cluster, including measures of how similar the documents in each cluster are.  This information is found in the file “cluster_desc.csv” in the “output_data_directory”. The “avg_centroid_distance” value measures the average distance of each document in the cluster from the cluster’s centroid. Similarly, the “intra_document_dispersion” value measures the average similarity of each document in the cluster to every other document in the cluster. The “dispersion_order”  attempts to re-arrange each cluster in an order that tries to reflect the inter document similarity. These statistics are informative, but they must be verified by human interpretation. They are a measure of how semantically similar documents are given the model’s training and the similarity of the portfolio—problems in the data can lead to problematic results.

The analysis will also tab each document with the corresponding cluster. This information is found in the file “cluster_master_labels.csv” in “output_data_directory”.