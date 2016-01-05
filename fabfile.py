from fabric.api import local

target = "w2v_pipeline/remove_parenthesis.py"

def push():
    local('git commit -a')
    local('git push')

def run():
    local("python {}".format(target))

def edit():
    local("emacs {} &".format(target))

def view():
    local("sqlitebrowser sql_data/1-2R01-2007.sqlite")

def test():
    clean()
    local("python w2v_pipeline/import_data.py")
    local("python w2v_pipeline/phrases_from_abbrs.py")
    local("python w2v_pipeline/replace_phrases.py")
    local("python w2v_pipeline/remove_parenthesis.py")

def clean():
    local('find . -name "*~" | xargs -I {} rm {}')
    local("rm -rf sql_data collated")

