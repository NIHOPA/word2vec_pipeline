from invoke import run as local
from invoke import task

package_dir = "word2vec_pipeline"

@task
def deploy(c=None):
    local("flake8 word2vec_pipeline")
    # local("nosetests -vs")
    # local("aspell check README.md")
    # local("check-manifest")


@task
def pep(c=None):
    local("autopep8 . -aaa --in-place --recursive --jobs=0")

@task
def import_data(c=None):
    local("python word2vec_pipeline import_data")

@task
def phrase(c=None):
    local("python word2vec_pipeline phrase")

@task
def parse(c=None):
    local("python word2vec_pipeline parse")

@task
def embed(c=None):
    local("python word2vec_pipeline embed")

@task
def score(c=None):
    local("python word2vec_pipeline score")

@task
def predict(c=None):
    local("python word2vec_pipeline predict")

@task
def cluster(c=None):
    local("python word2vec_pipeline cluster")

@task
def metacluster(c=None):
    local("python word2vec_pipeline metacluster")

@task
def analyze_metaclusters(c=None):
    local("python word2vec_pipeline analyze metacluster")

@task
def LIME(c=None):
    local("python word2vec_pipeline analyze LIME")

@task
def test(c=None):
    clean(c)

    import_data(c)
    phrase(c)
    parse(c)
    embed(c)
    score(c)

    metacluster(c)
    analyze_metaclusters(c)
    
    LIME(c)
    predict(c)

@task
def clean(c=None):
    local('find . -name "*~" | xargs -I {} rm {}')
    local('find . -name "*.pyc" | xargs -I {} rm {}')
    local('rm -rvf w2v.egg-info')
    local(
        "rm -rf data_import data_parsed data_document_scores "
        "data_clustering data_embeddings data_predict results")
