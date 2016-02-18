name = w2v_pipeline/parse.py

all:
	python $(name)
edit:
	emacs $(name) &
config:
	emacs config.ini &
