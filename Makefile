# --- Limpieza de artefactos y cache ---

SHELL := bash
PY := python
ENV ?= .env
include $(ENV)
export

print-%:
	@echo '$*=$($*)'

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
	@curl -s -H "Authorization: Bearer $(NC_ADMIN_TOKEN)" "http://$(API_HOST):$(API_PORT)/admin/cleanup/inventory?retention_days=$(NC_RETENTION_DAYS)&keep_last=$(NC_KEEP_LAST)" | jq .

admin-clean:
	@curl -s -X POST -H "Authorization: Bearer $(NC_ADMIN_TOKEN)" -H "Content-Type: application/json" -d "{\"retention_days\":$(NC_RETENTION_DAYS),\"keep_last\":$(NC_KEEP_LAST),\"dry_run\":false,\"force\":true,\"trash_dir\":\"$(NC_TRASH_DIR)\",\"trash_retention_days\":$(NC_TRASH_RETENTION_DAYS)}" "http://$(API_HOST):$(API_PORT)/admin/cleanup" | jq .
