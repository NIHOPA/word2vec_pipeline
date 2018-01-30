import numpy as np
from pygments import highlight, lexers, formatters

import sklearn.ensemble
import sklearn.preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics

import word2vec_pipeline.utils.db_utils as db
import word2vec_pipeline.utils.data_utils as uds
import word2vec_pipeline.document_scoring as ds
from lime.lime_text import LimeTextExplainer

import joblib, collections
import json, random
from tqdm import tqdm
import pandas as pd

def pp(js):
    s = json.dumps(js, indent=2)
    colorful_json = highlight(s,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print colorful_json


#print uds.load_document_vectors("unique_TF")
df = uds.load_ORG_data(["journal"])
LE = sklearn.preprocessing.LabelEncoder()
Y = LE.fit_transform(df.journal)
class_names = LE.classes_

explainer = LimeTextExplainer(class_names=class_names)
#help(explainer.explain_instance)


M = ds.score_unique()
def _vectorizer(text_blocks):
    return np.array([
        M.score_document({"text":text})['doc_vec']
        for text in text_blocks
    ])


vectorizer = sklearn.preprocessing.FunctionTransformer(_vectorizer, validate=False)

INPUT_ITR = db.item_iterator({}, text_column="text",)
ALL_TEXT = np.array([row['text'] for row in INPUT_ITR])
#print vectorizer.fit_transform(ALL_TEXT)


clf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=50,
    n_jobs=-1,
)
P = Pipeline([
    ('word2vec', vectorizer),
    ('randomforests', clf),
])

P.fit(ALL_TEXT, Y)
#proba = P.predict_proba(ALL_TEXT)
#print proba

i0, i1 = 0, 4000
sample1 = ALL_TEXT[i0]
sample2 = ALL_TEXT[i1]
print P.predict_proba([sample1,sample2])
l0, l1 = class_names[Y[i0]], class_names[Y[i1]],

def evaluate_text(text):
    exp = explainer.explain_instance(sample1, P.predict_proba, num_features=60)
    item = collections.Counter()
    for k,v in exp.as_list():
        item[k] += v
    return item
        

print "Starting"
random.shuffle(ALL_TEXT)
ITR = tqdm(ALL_TEXT[:])

data = collections.Counter()
func = joblib.delayed(evaluate_text)
with joblib.Parallel(-1) as MP:
    for res in MP(func(x) for x in ITR):
        data.update(res)
        
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
print df
df = df.sort_values(0)
print df
print class_names
df.to_csv("LIME_result.csv")

    
'''
exp = explainer.explain_instance(sample1, P.predict_proba, num_features=60)
pp(exp.as_list())
print l0, class_names[np.argmax(P.predict_proba([sample1,]))]

exp = explainer.explain_instance(sample2, P.predict_proba, num_features=60)
pp(exp.as_list())
print l1, class_names[np.argmax(P.predict_proba([sample2,]))]
'''


