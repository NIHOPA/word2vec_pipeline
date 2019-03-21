from fabric.api import local


def lint():
    local("black -l 80 fabfile.py pipeline_src")
    local("flake8 pipeline_src --ignore=E501,E203,W503,E402")


def import_data():
    local("python pipeline_src import_data")


def phrase():
    local("python pipeline_src phrase")


def parse():
    local("python pipeline_src parse")


def embed():
    local("python pipeline_src embed")


def score():
    local("python pipeline_src score")


def predict():
    local("python pipeline_src predict")


def cluster():
    local("python pipeline_src cluster")


def metacluster():
    local("python pipeline_src metacluster")


def analyze():
    local("python pipeline_src analyze")


def LIME():
    local("python pipeline_src analyze LIME")


def test():
    clean()

    import_data()
    phrase()
    parse()
    embed()
    score()

    metacluster()
    analyze()

    predict()


def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local('find . -name "*.pyc" | xargs -I {} rm {}')
    local("rm -rvf w2v.egg-info")
    local(
        "rm -rf data_import data_parsed data_document_scores "
        "data_clustering data_embeddings data_predict results"
    )
