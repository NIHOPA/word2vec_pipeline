from fabric.api import local

target = "w2v_pipeline/predict.py"

def push():
    local('git commit -a')
    local('git push')

def run():
    local("python {}".format(target))

def edit():
    local("emacs {} &".format(target))

def view():
    #local("sqlitebrowser data_sql/PLoS_bio.sqlite")
    local("sqlitebrowser data_parsed/PLoS_bio.sqlite")

def import_data():
    local("python w2v_pipeline/import_data.py")
    local("python w2v_pipeline/phrases_from_abbrs.py")  

def score():
    local("python w2v_pipeline/score.py")

def embed():
    local("python w2v_pipeline/embed.py")

def parse():
    local("python w2v_pipeline/parse.py")

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
    #cluster()
    metacluster()
    analyze_metaclusters()

def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local('find . -name "*.pyc" | xargs -I {} rm {}')
    local("rm -rf data_sql data_parsed data_document_scores data_clustering data_embeddings")

    

