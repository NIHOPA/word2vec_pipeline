from fabric.api import local

def push():
    local('git commit -a')
    local('git push')

def test():
    clean()
    local("python w2v_pipeline/import_data.py")
    local("python w2v_pipeline/phrases_from_abbrs.py")

def clean():
    local("rm -rf sql_data collated")
