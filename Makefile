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

upload:
	@echo "Uploading to PyPI..."
	@twine upload dist/*
	@echo "Finished uploading to PyPI..."

.PHONY: build