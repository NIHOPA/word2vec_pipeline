import lime
import word2vec_pipeline.utils.db_utils as db
import word2vec_pipeline.utils.data_utils as uds
import word2vec_pipeline.document_scoring as ds
#print uds.load_document_vectors("unique_TF")
#M = uds.load_w2vec()
M = ds.score_unique()

config = {"col":"text"}
INPUT_ITR = db.item_iterator(
    config,
    text_column="text",
    progress_bar=True,
    include_filename=True,
)

for row in INPUT_ITR:
    text = row['text']
    res = M.score_document(row)
    print res.keys()

