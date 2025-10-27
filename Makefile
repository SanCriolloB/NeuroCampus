# --- Limpieza de artefactos y cache ---

PY := python

ENV ?= .env
include $(ENV)
export

clean-inventory:
	@$(PY) -m tools.cleanup --inventory

clean-artifacts-dry-run:
	@$(PY) -m tools.cleanup --dry-run --retention-days $${NC_RETENTION_DAYS:-90} --keep-last $${NC_KEEP_LAST:-3}

# Para días posteriores (borrado real):
# clean-artifacts:
#	@$(PY) -m tools.cleanup --retention-days $${NC_RETENTION_DAYS:-90} --keep-last $${NC_KEEP_LAST:-3}
