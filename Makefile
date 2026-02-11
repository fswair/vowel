format:
	@echo "Formatting code with ruff..."
	@ruff format
	@echo "Running ruff checks and fixing issues..."
	@ruff check --fix --unsafe-fixes
	@echo "Type checking with ty..."
	@ty check
	@echo "Formatting complete."