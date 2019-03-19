from fabric.api import local

def lint():
    local(f"black -l 80 embedding_pipeline/")
    local(f"flake8 embedding_pipeline/ --ignore=E501,E203,W503")


def pep():
    local("autopep8 . -aaa --in-place --recursive --jobs=0")


def import_data():
    local("python embedding_pipeline/ import_data")


def phrase():
    local("python embedding_pipeline/ phrase")


def parse():
    local("python embedding_pipeline/ parse")


def embed():
    local("python embedding_pipeline/ embed")


def score():
    local("python embedding_pipeline/ score")


def predict():
    local("python embedding_pipeline/ predict")


def cluster():
    local("python embedding_pipeline/ cluster")


def metacluster():
    local("python embedding_pipeline/ metacluster")


def analyze_metaclusters():
    local("python embedding_pipeline/ analyze metacluster")


def LIME():
    local("python embedding_pipeline/ analyze LIME")


def test():
    clean()

    import_data()
    phrase()
    parse()
    embed()
    score()

    metacluster()
    analyze_metaclusters()
    
    LIME()
    predict()

def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local('find . -name "*.pyc" | xargs -I {} rm {}')
    local('rm -rvf w2v.egg-info')
    local(
        "rm -rf data_import data_parsed data_document_scores "
        "data_clustering data_embeddings data_predict results")
