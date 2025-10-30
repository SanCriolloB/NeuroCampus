# --- NeuroCampus: Limpieza de artefactos, administración y diagnóstico Día 5 ---

# Usar bash para comandos con tuberías/expansiones
SHELL := bash

# Binario de Python
PY := python

# ==== VENV aware (Windows / POSIX) ====
VENV ?= .venv

ifeq ($(OS),Windows_NT)
PY     := $(VENV)/Scripts/python.exe
PIP    := $(VENV)/Scripts/pip.exe
PYTEST := $(VENV)/Scripts/pytest.exe
else
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PYTEST := $(PY) -m pytest
endif

# Archivo de variables de entorno (puede redefinir API_HOST, API_PORT, etc.)
ENV ?= .env
-include $(ENV)
export

# Valores por defecto (si no vienen de .env)
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
NC_RETENTION_DAYS ?= 90
NC_KEEP_LAST ?= 3
NC_EXCLUDE_GLOBS ?=
NC_TRASH_DIR ?= .trash
NC_TRASH_RETENTION_DAYS ?= 7
NC_ADMIN_TOKEN ?= dev-admin-token

# Para el helper de validación
NC_DATASET_ID ?= docentes
NC_SAMPLE_CSV ?= examples/docentes.csv
NC_FMT ?=

# Rutas comunes
BACKEND_SRC ?= backend/src
BACKEND_APP ?= neurocampus.app.main:app
FRONTEND_DIR ?= frontend

# ----------------------------------------------------------------------------- #
# Objetivos
# ----------------------------------------------------------------------------- #

.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  clean-inventory             - Ver inventario local de artefactos (herramienta CLI)."
	@echo "  clean-artifacts-dry-run     - Simulación de limpieza local (no borra, muestra plan)."
	@echo "  clean-artifacts             - Limpieza local real (mueve a .trash/)."
	@echo "  run-admin                   - Levantar API con routers (modo admin/desarrollo)."
	@echo "  admin-inventory             - Inventario remoto vía /admin/cleanup/inventory."
	@echo "  admin-clean                 - Limpieza remota vía /admin/cleanup (force)."
	@echo "  be-dev                      - Levantar backend en modo desarrollo (uvicorn)."
	@echo "  be-test                     - Ejecutar pruebas del backend (pytest)."
	@echo "  fe-dev                      - Levantar frontend (Vite) en :5173."
	@echo "  fe-build                    - Construir frontend (Vite build)."
	@echo "  fe-preview                  - Previsualizar build del FE en :4173."
	@echo "  fe-test                     - Ejecutar pruebas del frontend (vitest)."
	@echo "  fe-typecheck                - Chequeo de tipos del FE (tsc --noEmit)."
	@echo "  validate-sample             - Enviar CSV de ejemplo a /datos/validar."
	@echo "  validate-sample-fmt         - Igual que arriba pero forzando formato (NC_FMT=csv|xlsx|parquet)."
	@echo ""
	@echo "Variables (.env o CLI):"
	@echo "  API_HOST, API_PORT, NC_ADMIN_TOKEN, NC_RETENTION_DAYS, NC_KEEP_LAST"
	@echo "  NC_TRASH_DIR, NC_TRASH_RETENTION_DAYS, NC_DATASET_ID, NC_SAMPLE_CSV, NC_FMT"

# ----------------------------------------------------------------------------- #
# --- Limpieza de artefactos y cache (local, vía herramienta CLI) -------------- #
# ----------------------------------------------------------------------------- #

.PHONY: clean-inventory
clean-inventory:
	@$(PY) -m tools.cleanup --inventory

.PHONY: clean-artifacts-dry-run
clean-artifacts-dry-run:
	@$(PY) -m tools.cleanup --dry-run \
		--retention-days $${NC_RETENTION_DAYS:-90} \
		--keep-last $${NC_KEEP_LAST:-3} \
		--exclude-globs "$${NC_EXCLUDE_GLOBS:-}"

# Borrado real (mueve a .trash/). Requiere --force.
.PHONY: clean-artifacts
clean-artifacts:
	@$(PY) -m tools.cleanup --force \
		--retention-days $${NC_RETENTION_DAYS:-90} \
		--keep-last $${NC_KEEP_LAST:-3} \
		--exclude-globs "$${NC_EXCLUDE_GLOBS:-}" \
		--trash-dir "$${NC_TRASH_DIR:-.trash}"

# ----------------------------------------------------------------------------- #
# --- Administración (vía API) ------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Levanta FastAPI/uvicorn apuntando al directorio del backend.
.PHONY: run-admin
run-admin:
	@uvicorn $(BACKEND_APP) --app-dir $(BACKEND_SRC) --host $${API_HOST} --port $${API_PORT} --reload

# Consulta inventario remoto del endpoint de administración.
.PHONY: admin-inventory
admin-inventory:
	@curl -s \
		-H "Authorization: Bearer $(NC_ADMIN_TOKEN)" \
		"http://$(API_HOST):$(API_PORT)/admin/cleanup/inventory?retention_days=$(NC_RETENTION_DAYS)&keep_last=$(NC_KEEP_LAST)" | jq .

# Ejecuta limpieza remota en el backend (force=true).
.PHONY: admin-clean
admin-clean:
	@curl -s -X POST \
		-H "Authorization: Bearer $(NC_ADMIN_TOKEN)" \
		-H "Content-Type: application/json" \
		-d "{\"retention_days\":$(NC_RETENTION_DAYS),\"keep_last\":$(NC_KEEP_LAST),\"dry_run\":false,\"force\":true,\"trash_dir\":\"$(NC_TRASH_DIR)\",\"trash_retention_days\":$(NC_TRASH_RETENTION_DAYS)}" \
		"http://$(API_HOST):$(API_PORT)/admin/cleanup" | jq .

# ----------------------------------------------------------------------------- #
# --- Backend (desarrollo y pruebas) ------------------------------------------ #
# ----------------------------------------------------------------------------- #

# Levantar backend de desarrollo (equivalente a run-admin; alias más neutral).
.PHONY: be-dev
be-dev:
	@uvicorn $(BACKEND_APP) --app-dir $(BACKEND_SRC) --host $${API_HOST} --port $${API_PORT} --reload

# Ejecutar pruebas del backend. Forzamos PYTHONPATH para resolver imports de backend/src.
.PHONY: be-test
be-test:
	@PYTHONPATH=$(BACKEND_SRC) $(PYTEST) -q

# Opcional: ejecutar tests desactivando auth admin (útil para depurar)
.PHONY: be-test-noauth
be-test-noauth:
	@NC_DISABLE_ADMIN_AUTH=1 PYTHONPATH=$(BACKEND_SRC) $(PYTEST) -q


# ----------------------------------------------------------------------------- #
# --- Frontend (desarrollo, build y pruebas) ---------------------------------- #
# ----------------------------------------------------------------------------- #

.PHONY: fe-dev
fe-dev:
	@cd $(FRONTEND_DIR) && npm run dev

.PHONY: fe-build
fe-build:
	@cd $(FRONTEND_DIR) && npm run build

.PHONY: fe-preview
fe-preview:
	@cd $(FRONTEND_DIR) && npm run preview

.PHONY: fe-test
fe-test:
	@cd $(FRONTEND_DIR) && npm run test:run

.PHONY: fe-typecheck
fe-typecheck:
	@cd $(FRONTEND_DIR) && npx tsc --noEmit

# ----------------------------------------------------------------------------- #
# --- Diagnóstico Día 5: validación de datasets ------------------------------- #
# ----------------------------------------------------------------------------- #

# Helper: envía un CSV de ejemplo al endpoint /datos/validar del backend.
# Variables:
#   - NC_SAMPLE_CSV (ruta del archivo CSV a enviar)
#   - NC_DATASET_ID (identificador lógico del dataset, por defecto "docentes")
.PHONY: validate-sample
validate-sample:
	@test -f "$(NC_SAMPLE_CSV)" || (echo "ERROR: No existe $(NC_SAMPLE_CSV). Ajusta NC_SAMPLE_CSV o agrega un ejemplo." && exit 1)
	@echo ">> Validando archivo '$(NC_SAMPLE_CSV)' como dataset_id='$(NC_DATASET_ID)' contra http://$(API_HOST):$(API_PORT)/datos/validar"
	@curl -s -F "file=@$(NC_SAMPLE_CSV)" -F "dataset_id=$(NC_DATASET_ID)" \
		"http://$(API_HOST):$(API_PORT)/datos/validar" | jq .

# Igual que validate-sample pero permitiendo forzar el lector con NC_FMT=csv|xlsx|parquet
.PHONY: validate-sample-fmt
validate-sample-fmt:
	@test -f "$(NC_SAMPLE_CSV)" || (echo "ERROR: No existe $(NC_SAMPLE_CSV). Ajusta NC_SAMPLE_CSV o agrega un ejemplo." && exit 1)
	@test -n "$(NC_FMT)" || (echo "ERROR: Define NC_FMT=csv|xlsx|parquet" && exit 1)
	@echo ">> Validando archivo '$(NC_SAMPLE_CSV)' como dataset_id='$(NC_DATASET_ID)' (fmt=$(NC_FMT)) contra http://$(API_HOST):$(API_PORT)/datos/validar"
	@curl -s -F "file=@$(NC_SAMPLE_CSV)" -F "dataset_id=$(NC_DATASET_ID)" -F "fmt=$(NC_FMT)" \
		"http://$(API_HOST):$(API_PORT)/datos/validar" | jq .


# Activar el entorno virtual

.PHONY: venv
venv:
	@python -m venv .venv && echo ">> Activa el entorno: source .venv/Scripts/activate"

.PHONY: deps-be
deps-be:
	@$(PIP) install -U pip wheel setuptools
	@$(PIP) install -r backend/requirements.txt