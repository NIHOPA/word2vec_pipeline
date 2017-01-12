name = w2v_pipeline/parse.py

edit:
	emacs $(name) &
config:
	emacs config.ini &


fix_errors = "W293,W291,W391,W231,W235,E231,E302,E303,E221,E225,E265,E225,E221,F401,E203,E401,E124,E202,E201,E211"

flake_clean:
	autopep8 -vvv -a --in-place --recursive --jobs 0  .
#	#--select=$(fix_errors) .

flake:
	flake8 -j 'auto' . 
