from fabric.api import local

package_dir = "word2vec_pipeline"

def deploy():
    #local("nosetests -vs")
    local("flake8 --ignore=E501,F821 w2v_pipeline")
    local("aspell check README.md")
    local("check-manifest")
    #local("python miniprez tutorial.md")

def pep():
    local("autopep8 {}/*.py -a --in-place --jobs=0".format(package_dir))

def import_data():
    local("python {}/import_data.py".format(package_dir))
    local("python {}/phrases_from_abbrs.py".format(package_dir))


def parse():
    local("python {}/parse.py".format(package_dir))


def embed():
    local("python {}/embed.py".format(package_dir))


def score():
    local("python {}/score.py".format(package_dir))


def predict():
    local("python {}/predict.py".format(package_dir))


def cluster():
    local("python {}/cluster.py".format(package_dir))


def metacluster():
    local("python {}/metacluster.py".format(package_dir))


def analyze_metaclusters():
    cmd = "python {}/postprocessing/analyze_metaclusters.py"
    local(cmd.format(package_dir))


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
    local('rm -rvf w2v.egg-info')
    local(
        "rm -rf data_import data_parsed data_document_scores data_clustering data_embeddings data_predict results")
