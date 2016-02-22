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

def _import():
    local("python w2v_pipeline/import.py")
    local("python w2v_pipeline/phrases_from_abbrs.py")  

def compute():
    local("python w2v_pipeline/compute.py")

def parse():
    local("python w2v_pipeline/parse.py")

def test():
    clean()
    
    _import()
    parse()
    compute()
    
    #local("python w2v_pipeline/compute_kSVD.py")

def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local("rm -rf data_sql data_parsed collated")

    

