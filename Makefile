build:
	@echo "Cleaning build..."
	@rm -rf dist build
	@echo "Cleaned dist and build..."
	@echo "Building wheel..."
	@pip install wheel
	@echo "Build finished..."
	@echo "Making distributions..."
	@python setup.py bdist_wheel sdist
	@echo "Finished making distributions..."

upload: check
	@echo "Uploading to PyPI..."
	@twine upload dist/*
	@echo "Finished uploading to PyPI..."

check:
	@echo "Checking code..."
	@black --check mlflow_extensions/
	@black --check scripts/
	@echo "Finished checking code..."

fmt:
	@echo "Formatting code..."
	@black mlflow_extensions/
	@black scripts/
	@echo "Finished formatting code..."

.PHONY: build