PYTHON :=  $(shell which python)

TOP_DIR := .
SRC_DIR := $(TOP_DIR)/mlflow_extensions
DIST_DIR := $(TOP_DIR)/dist
TEST_DIR := $(TOP_DIR)/tests
SCRIPTS_DIR := $(TOP_DIR)/scripts
LIB_NAME := mlflow_extensions
LIB_VERSION := $(shell $(PYTHON) -c "from mlflow_extensions.version import get_mlflow_extensions_version; print(get_mlflow_extensions_version())")
LIB_SUFFIX := py3-none-any.whl
LIB := $(LIB_NAME)-$(LIB_VERSION)-$(LIB_SUFFIX)
TARGET := $(DIST_DIR)/$(LIB)


BDIST := $(PYTHON) setup.py bdist_wheel sdist
PIP_INSTALL := $(PYTHON) -m pip install 
PYTEST := $(shell which pytest) -s
BLACK := $(shell which black)
ISORT := $(shell which isort)
PUBLISH := $(shell which twine) upload
FIND := $(shell which find)
RM := rm -rf
CD := cd

all: build

build: 
	@echo "Building: $(LIB)..."
	@$(PIP_INSTALL) wheel
	@$(BDIST)
	@echo "Finished building: $(LIB)."

upload: check
	@echo "Uploading to PyPI..."
	@$(PUBLISH) upload dist/*
	@echo "Finished uploading to PyPI."

check:
	@echo "Checking code..."
	@$(BLACK) --check  $(SRC_DIR) $(SCRIPTS_DIR) $(TEST_DIR)
	@echo "Finished checking code."

fmt:
	@echo "Formatting code..."
	@$(BLACK) $(SRC_DIR) $(SCRIPTS_DIR) $(TEST_DIR)
	@$(ISORT) $(SRC_DIR) $(SCRIPTS_DIR) $(TEST_DIR)
	@echo "Finished formatting code."

clean:
	@echo "Cleaning up intermediate artifacts..."
	@$(FIND) $(SRC_DIR) -name \*.pyc -exec rm -f {} \;
	@$(FIND) $(SRC_DIR) -name \*.pyo -exec rm -f {} \;
	@$(FIND) $(TEST_DIR) -name \*.pyc -exec rm -f {} \;
	@$(FIND) $(TEST_DIR) -name \*.pyo -exec rm -f {} \;
	@$(FIND) $(SCRIPTS_DIR) -name \*.pyc -exec rm -f {} \;
	@$(FIND) $(SCRIPTS_DIR) -name \*.pyo -exec rm -f {} \;
	@echo "Finishing cleaning up intermediate artifacts."

distclean: clean
	@echo "Cleaning up distribution artifacts..."
	@$(RM) $(DIST_DIR)
	@$(RM) $(SRC_DIR)/*.egg-info
	@$(RM) $(TOP_DIR)/.mypy_cache
	@$(FIND) $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR) \( -name __pycache__ -a -type d \) -prune -exec rm -rf {} \;
	@echo "Finished cleaning up distribution artifacts."

test:
	$(PYTEST) $(TEST_DIR)

help:
	$(info TOP_DIR: $(TOP_DIR))
	$(info SRC_DIR: $(SRC_DIR))
	$(info DIST_DIR: $(DIST_DIR))
	$(info TEST_DIR: $(TEST_DIR))
	$(info SCRIPTS_DIR: $(SCRIPTS_DIR))
	$(info LIB: $(LIB))
	$(info )
	$(info $$> make [all|build|clean|distclean|fmt|check|test|upload])
	$(info )
	$(info       all          - build library: [$(LIB)]. This is the default)
	$(info       build        - build library: [$(LIB)])
	$(info       clean        - removes build artifacts)
	$(info       distclean    - removes distribution artifacts)
	$(info       fmt          - format source code)
	$(info       check        - format source code)
	$(info       test         - run unit tests)
	$(info       upload       - publish wheel to pypi/artifactory server)

	@true

.PHONY: build upload check fmt clean distclean help
