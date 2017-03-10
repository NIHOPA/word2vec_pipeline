name = word2vec_pipeline/parse.py

edit:
	emacs $(name) &
config:
	emacs config.ini &

flake_clean:
	autopep8 -aaaa -v --in-place --recursive --jobs 0  word2vec_pipeline

flake:
	flake8 --ignore F821 -j 'auto' word2vec_pipeline/ 

future:
# Fix print statements
	2to3 -n -w -j 8 --fix print .
# Fix unneeded imports and variables
	find . -name "*.py" | xargs -I {} autoflake {} --in-place --remove-unused-variables
