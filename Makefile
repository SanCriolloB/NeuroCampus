# --- NeuroCampus: Limpieza de artefactos, administración y diagnóstico Día 7 ---

# Usar bash para comandos con tuberías/expansiones
SHELL := bash

# Binario de Python
PY := python

# Archivo de variables de entorno (puede redefinir API_HOST, API_PORT, etc.)
ENV ?= .env
-include $(ENV)
export

# Valores por defecto (si no vienen de .env)
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

# Administración de limpieza
NC_RETENTION_DAYS ?= 90
NC_KEEP_LAST ?= 3
NC_EXCLUDE_GLOBS ?=
NC_TRASH_DIR ?= .trash
NC_TRASH_RETENTION_DAYS ?= 7
NC_ADMIN_TOKEN ?= dev-admin-token

# Validación datasets (helper)
NC_DATASET_ID ?= docentes
NC_SAMPLE_CSV ?= examples/docentes.csv

# CORS / Límite de subida
# - NC_ALLOWED_ORIGINS se usa en main.py (middleware CORS)
# - NC_MAX_UPLOAD_MB se usa en main.py (middleware) y también aquí para uvicorn
NC_ALLOWED_ORIGINS ?= http://localhost:5173
NC_MAX_UPLOAD_MB ?= 10

# Rutas comunes
BACKEND_SRC ?= backend/src
BACKEND_APP ?= neurocampus.app.main:app
FRONTEND_DIR ?= frontend

# ----------------------------------------------------------------------------- #
# Ayuda
# ----------------------------------------------------------------------------- #

.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  clean-inventory             - Ver inventario local de artefactos (herramienta CLI)."
	@echo "  clean-artifacts-dry-run     - Simulación de limpieza local (no borra, muestra plan)."
	@echo "  clean-artifacts             - Limpieza local real (mueve a .trash/)."
	@echo "  run-admin                   - Levanta API (uvicorn) con routers de administración."
	@echo "  admin-inventory             - Inventario remoto vía endpoint /admin/cleanup/inventory."
	@echo "  admin-clean                 - Limpieza remota vía endpoint /admin/cleanup (force)."
	@echo "  be-dev                      - Levantar backend en modo desarrollo (uvicorn)."
	@echo "  be-test                     - Ejecutar pruebas del backend (pytest)."
	@echo "  fe-dev                      - Levantar frontend (Vite) en :5173."
	@echo "  fe-test                     - Ejecutar pruebas del frontend (vitest)."
	@echo "  validate-sample             - Enviar CSV de ejemplo a /datos/validar."
	@echo "  rbm-audit                   - Ejecutar auditoría K-Fold (YAML en la raíz)."
	@echo "  rbm-pura-audit              - Alias de rbm-audit (foco en rbm_pura en CI/logs)."
	@echo "  rbm-general-audit           - Alias de rbm-audit (foco en rbm_general en CI/logs)."
	@echo "  rbm-restringido-audit       - Alias de rbm-audit (foco en rbm_restringido en CI/logs)."
	@echo "  sim-data                    - Genera dataset sintético compatible."
	@echo "  train-rbm-pura              - Entrena RBM pura (script CLI)."
	@echo "  train-rbm-general           - Entrena RBM general (script CLI)."
	@echo
	@echo "Variables útiles (pueden ir en .env o CLI):"
	@echo "  API_HOST, API_PORT, NC_ADMIN_TOKEN"
	@echo "  NC_RETENTION_DAYS, NC_KEEP_LAST, NC_EXCLUDE_GLOBS"
	@echo "  NC_TRASH_DIR, NC_TRASH_RETENTION_DAYS"
	@echo "  NC_DATASET_ID, NC_SAMPLE_CSV"
	@echo "  NC_ALLOWED_ORIGINS, NC_MAX_UPLOAD_MB"
	@echo "  BACKEND_SRC, BACKEND_APP, FRONTEND_DIR"

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
# Defensa adicional Día 7: límite de tamaño en uvicorn en BYTES (MB * 1024 * 1024).
.PHONY: run-admin
run-admin:
	@echo ">> Iniciando uvicorn con límite máx. de request: $${NC_MAX_UPLOAD_MB:-10} MB"
	@uvicorn $(BACKEND_APP) --app-dir $(BACKEND_SRC) \
		--host $${API_HOST} --port $${API_PORT} --reload \
		--limit-max-request-size $$(( $${NC_MAX_UPLOAD_MB:-10} * 1024 * 1024 ))

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
# El límite de tamaño se aplica vía middleware (NC_MAX_UPLOAD_MB), no por flag de Uvicorn.
.PHONY: be-dev
be-dev:
	@echo ">> Iniciando uvicorn (desarrollo). Límite de subida via middleware: $${NC_MAX_UPLOAD_MB:-10} MB"
	@uvicorn $(BACKEND_APP) --app-dir $(BACKEND_SRC) \
		--host $${API_HOST} --port $${API_PORT} --reload

# Ejecutar pruebas del backend. Forzamos PYTHONPATH para resolver imports de backend/src.
.PHONY: be-test
be-test:
	@PYTHONPATH=$(BACKEND_SRC) $(PY) -m pytest -q

# ----------------------------------------------------------------------------- #
# --- Frontend (desarrollo y pruebas) ----------------------------------------- #
# ----------------------------------------------------------------------------- #

.PHONY: fe-dev
fe-dev:
	@cd $(FRONTEND_DIR) && npm run dev

.PHONY: fe-test
fe-test:
	@cd $(FRONTEND_DIR) && npm run test:run

# ----------------------------------------------------------------------------- #
# --- Diagnóstico: validación de datasets ------------------------------------- #
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

# ----------------------------------------------------------------------------- #
# --- Auditoría RBM (K-Fold) -------------------------------------------------- #
# ----------------------------------------------------------------------------- #

.PHONY: rbm-audit
rbm-audit:
	@PY=$$( [ -x ".venv/Scripts/python.exe" ] && echo ".venv/Scripts/python.exe" || ( [ -x ".venv/bin/python" ] && echo ".venv/bin/python" || echo "python" ) ); \
	echo "Usando Python: $$PY"; \
	"$$PY" -c "import numpy" 2>/dev/null || ( echo "Instalando deps en $$PY"; "$$PY" -m pip install --upgrade pip && "$$PY" -m pip install -r backend/requirements.txt ); \
	PPATH="$(PWD)/backend/src"; echo "PYTHONPATH: $$PPATH"; \
	PYTHONPATH="$$PPATH" "$$PY" -m neurocampus.models.audit_kfold --config configs/rbm_audit.yaml

# ----------------------------------------------------------------------------- #
# --- Auditorías por modelo (alias de conveniencia) --------------------------- #
# ----------------------------------------------------------------------------- #

.PHONY: rbm-pura-audit rbm-general-audit rbm-restringido-audit

# Nota: estos targets son alias y reutilizan el mismo YAML (que incluye los 3 modelos).
rbm-pura-audit:
	@$(MAKE) rbm-audit

rbm-general-audit:
	@$(MAKE) rbm-audit

rbm-restringido-audit:
	@$(MAKE) rbm-audit

# ----------------------------------------------------------------------------- #
# --- Simulación de datos y entrenamiento directo ----------------------------- #
# ----------------------------------------------------------------------------- #

.PHONY: sim-data train-rbm-pura train-rbm-general

# Genera dataset sintético coherente con el pipeline (calif_*, p_*, label teacher)
sim-data:
	@$(PY) tools/sim/generate_synthetic.py --n 5000 --out data/simulated/evals_sim_5k.parquet

# Entrena RBM Pura de forma directa (no supervisada)
train-rbm-pura:
	@PYTHONPATH=$(BACKEND_SRC) $(PY) backend/scripts/train_rbm_pura.py \
		--data data/labeled/evaluaciones_2025_teacher.parquet

# Entrena RBM General de forma directa (supervisada mínima)
train-rbm-general:
	@PYTHONPATH=$(BACKEND_SRC) $(PY) backend/scripts/train_rbm_general.py \
		--data data/labeled/evaluaciones_2025_teacher.parquet \
		--target sentiment_label_teacher
