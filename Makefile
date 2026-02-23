BLUE := \033[1;34m
GREEN := \033[1;32m
RESET := \033[0m

format:
	@printf "$(BLUE)==>$(RESET) Formatting code with ruff...\n"
	@ruff format
	@printf "$(GREEN)✔ Formatting complete.$(RESET)\n"

check:
	@printf "$(BLUE)==>$(RESET) Running ruff checks and fixing issues...\n"
	@ruff check --fix --unsafe-fixes
	@printf "$(BLUE)==>$(RESET) Type checking with ty...\n"
	@ty check
	@printf "$(BLUE)==>$(RESET) Type checking with basedpyright...\n"
	@basedpyright
	@printf "$(GREEN)✔ Checking complete.$(RESET)\n"

all: format check