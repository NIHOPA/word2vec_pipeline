name = w2v_pipeline/phrases_from_abbrs.py

all:
	python $(name)
edit:
	emacs $(name) &
config:
	emacs config.ini &
