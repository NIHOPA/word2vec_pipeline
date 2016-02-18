from fabric.api import local

target = "w2v_pipeline/compute_scores.py -f"

def push():
    local('git commit -a')
    local('git push')

def run():
    local("python {}".format(target))

def edit():
    local("emacs {} &".format(target))

def view():
    local("sqlitebrowser sql_data/PLoS_bio.sqlite")

def test():
    clean()
    local("python w2v_pipeline/import_data.py")
    local("python w2v_pipeline/phrases_from_abbrs.py")
    local("python w2v_pipeline/replace_phrases.py")
    local("python w2v_pipeline/remove_parenthesis.py")
    local("python w2v_pipeline/token_replacement.py")
    local("python w2v_pipeline/decaps_text.py")
    local("python w2v_pipeline/pos_tokenizer.py")
    local("python w2v_pipeline/compute_TF.py")
    local("python w2v_pipeline/compute_features.py")
    local("python w2v_pipeline/compute_kSVD.py")
    local("python w2v_pipeline/compute_scores.py")

def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local("rm -rf sql_data collated models")

