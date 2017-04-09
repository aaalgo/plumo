all:
	git submodule init
	git submodule update
	nvidia-docker build -t aaalgo/plumo .
