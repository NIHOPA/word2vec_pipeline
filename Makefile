name = w2v_pipeline/import.py

all:
	python $(name)
edit:
	emacs $(name) &
config:
	emacs config.ini &
