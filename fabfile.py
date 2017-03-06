from fabric.api import local

def deploy():
    #local("nosetests -vs")
    local("flake8 --ignore=E501,F821 w2v_pipeline")
    local("aspell check README.md")
    local("check-manifest")
    #local("python miniprez tutorial.md")

def pep():
    local("autopep8 w2v_pipeline/*.py -a --in-place --jobs=0")

def import_data():
    local("python w2v_pipeline/import_data.py")
    local("python w2v_pipeline/phrases_from_abbrs.py")


def parse():
    local("python w2v_pipeline/parse.py")


def embed():
    local("python w2v_pipeline/embed.py")


def score():
    local("python w2v_pipeline/score.py")


def predict():
    local("python w2v_pipeline/predict.py")


def cluster():
    local("python w2v_pipeline/cluster.py")


def metacluster():
    local("python w2v_pipeline/metacluster.py")


def analyze_metaclusters():
    local("python w2v_pipeline/postprocessing/analyze_metaclusters.py")


def test():
    clean()

    import_data()
    parse()
    embed()
    score()
    predict()
    # cluster()
    metacluster()
    analyze_metaclusters()


def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local('find . -name "*.pyc" | xargs -I {} rm {}')
    local(
        "rm -rf data_import data_parsed data_document_scores data_clustering data_embeddings data_predict results")
