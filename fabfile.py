from fabric.api import local

def push():
    local('git commit -a')
