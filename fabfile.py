from fabric.api import local

package_dir = "word2vec_pipeline"


def deploy():
    local("flake8 --ignore=E501,F821 word2vec_pipeline tests")
    # local("nosetests -vs")
    # local("aspell check README.md")
    # local("check-manifest")


def pep():
    local("autopep8 . -aaa --in-place --recursive --jobs=0")


def import_data():
    local("python word2vec_pipeline import_data")


def phrase():
    local("python word2vec_pipeline phrase")


def parse():
    local("python word2vec_pipeline parse")


def embed():
    local("python word2vec_pipeline embed")


def score():
    local("python word2vec_pipeline score")


def predict():
    local("python word2vec_pipeline predict")


def cluster():
    local("python word2vec_pipeline cluster")


def metacluster():
    local("python word2vec_pipeline metacluster")


def analyze_metaclusters():
    local("python word2vec_pipeline analyze metacluster")

def LIME():
    local("python word2vec_pipeline analyze LIME")


def test():
    clean()

    import_data()
    phrase()
    parse()
    embed()
    score()

    metacluster()
    analyze_metaclusters()
    predict()

    # cluster()


def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local('find . -name "*.pyc" | xargs -I {} rm {}')
    local('rm -rvf w2v.egg-info')
    local(
        "rm -rf data_import data_parsed data_document_scores "
        "data_clustering data_embeddings data_predict results")
