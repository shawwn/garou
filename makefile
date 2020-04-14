.PHONY: all clean test

PYMEN := bin/pymen

MODS := resnet.x

all: $(MODS:.x=.py)

clean:
	@git checkout *.py

%.py : %.l
	@echo $@
	@$(PYMEN) -c $< -o $@ -t py

test: all
	@echo py:
	@$(PYMEN) ./test.l
