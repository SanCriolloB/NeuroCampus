# --- Limpieza de artefactos y cache ---

PY := python
ENV ?= .env
include $(ENV)
export

clean-inventory:
	@$(PY) -m tools.cleanup --inventory

clean-artifacts-dry-run:
	@$(PY) -m tools.cleanup --dry-run --retention-days $${NC_RETENTION_DAYS:-90} --keep-last $${NC_KEEP_LAST:-3} --exclude-globs "$${NC_EXCLUDE_GLOBS:-}"

# Borrado real (mueve a .trash/). Requiere --force.
clean-artifacts:
	@$(PY) -m tools.cleanup --force --retention-days $${NC_RETENTION_DAYS:-90} --keep-last $${NC_KEEP_LAST:-3} --exclude-globs "$${NC_EXCLUDE_GLOBS:-}" --trash-dir "$${NC_TRASH_DIR:-.trash}"

run-admin:
	@uvicorn neurocampus.app.main:app --app-dir backend/src --host $${API_HOST} --port $${API_PORT} --reload

admin-inventory:
 @curl -H "Authorization: Bearer $(NC_ADMIN_TOKEN)" \
   "http://$(API_HOST):$(API_PORT)/admin/cleanup/inventory?retention_days=$(NC_RETENTION_DAYS)&keep_last=$(NC_KEEP_LAST)"