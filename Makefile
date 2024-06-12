#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cifar_mlrf
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 MLRF
	isort --check --diff --profile black MLRF
	black --check --config pyproject.toml MLRF

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml MLRF




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	pipenv --python $(PYTHON_VERSION)
	@echo ">>> New pipenv created. Activate with:\npipenv shell"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) MLRF/dataset.py

## Train Model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) MLRF/modeling/train.py

## Predict Model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) MLRF/modeling/predict.py

## Plots
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) MLRF/plots.py

## Run the entire pipeline
.PHONY: run
run: requirements data train predict


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
