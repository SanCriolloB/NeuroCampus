# ===========================
# Makefile - NeuroCampus (simple y robusto)
# ===========================

# -------- Detección robusta de PYTHON --------
ifeq ($(OS),Windows_NT)
PY_BACKEND := ./backend/.venv/Scripts/python.exe
PY_ROOT    := ./.venv/Scripts/python.exe
PATHSEP    := ;
else
PY_BACKEND := ./backend/.venv/bin/python
PY_ROOT    := ./.venv/bin/python
PATHSEP    := :
endif

PYTHON ?= $(shell \
	if [ -x "$(PY_BACKEND)" ]; then echo "$(PY_BACKEND)"; \
	elif [ -x "$(PY_ROOT)" ]; then echo "$(PY_ROOT)"; \
	else which python; fi \
)

# Rutas y variables comunes (RELATIVAS para que funcionen bien en Windows)
SRC_DIR   := backend/src
EXAMPLES  := examples
OUT_DIR   := data/prep_auto

# -------- Variables del pipeline --------
BETO_MODE       ?= simple
MIN_TOKENS      ?= 3
MAX_TOKENS      ?= 256
VAL_SIZE        ?= 0.2
TEST_SIZE       ?= 0.1
RANDOM_STATE    ?= 42
BATCH_SIZE      ?= 32
THRESHOLD       ?= 0.45
MARGIN          ?= 0.05
NEU_MIN         ?= 0.10
TEXT_COLS       ?= auto

# ===========================
# Entorno backend
# ===========================

.PHONY: which-python
which-python:
	@echo "PY_BACKEND = $(PY_BACKEND)"
	@echo "PY_ROOT    = $(PY_ROOT)"
	@echo "PYTHON     = $(PYTHON)"

# Crear venv SOLO para backend (backend/.venv)
.PHONY: venv-backend-create
venv-backend-create:
	@cd backend && python -m venv .venv
	@echo ">> Activa el entorno con:"
	@echo "   Windows: backend\\.venv\\Scripts\\activate"
	@echo "   Unix:    source backend/.venv/bin/activate"

# Instalar requirements del backend en backend/.venv
.PHONY: install-reqs
install-reqs:
	@test -x "$(PY_BACKEND)" || (echo "ERROR: no se encontró backend/.venv. Ejecuta 'make venv-backend-create' antes." && exit 1)
	@cd backend && .venv/Scripts/pip.exe install --upgrade pip || .venv/bin/pip install --upgrade pip
	@cd backend && .venv/Scripts/pip.exe install -r requirements.txt || .venv/bin/pip install -r requirements.txt

# ===========================
# Preproceso - Paso 1
# ===========================

# NOTA: usamos los scripts reales:
#   - neurocampus.app.jobs.cmd_preprocesar_beto        (un solo dataset)
#   - neurocampus.app.jobs.cmd_preprocesar_batch       (varios datasets)
#   - neurocampus.app.jobs.validate_prep_dir           (validación)

.PHONY: prep-one
prep-one:
	@mkdir -p "$(OUT_DIR)"
	@echo "[prep-one] Usando BETO_MODE=$(BETO_MODE), entrada $(NC_SAMPLE_CSV), salida en $(OUT_DIR)"
	@test -f "$(NC_SAMPLE_CSV)" || (echo "ERROR: No existe $(NC_SAMPLE_CSV). Ajusta NC_SAMPLE_CSV." && exit 1)
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_beto \
		--in "$(NC_SAMPLE_CSV)" \
		--out "$(OUT_DIR)/$(NC_DATASET_ID).parquet" \
		--text-col "$(TEXT_COLS)" \
		--beto-mode "$(BETO_MODE)" \
		--batch-size "$(BATCH_SIZE)" \
		--threshold "$(THRESHOLD)" \
		--margin "$(MARGIN)" \
		--neu-min "$(NEU_MIN)"

# Preprocesar todos los datasets encontrados en examples/ y examples/synthetic
.PHONY: prep-all
prep-all:
	@mkdir -p "$(OUT_DIR)"
	@echo "[prep-all] Preprocesando CSV en $(EXAMPLES) y $(EXAMPLES)/synthetic → $(OUT_DIR)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_batch \
		--in-dirs "$(EXAMPLES),$(EXAMPLES)/synthetic" \
		--out-dir "$(OUT_DIR)" \
		--text-cols "$(TEXT_COLS)" \
		--beto-mode "$(BETO_MODE)" \
		--min-tokens "$(MIN_TOKENS)" \
		--threshold "$(THRESHOLD)" \
		--margin "$(MARGIN)" \
		--neu-min "$(NEU_MIN)"

# ===========================
# Preproceso - Limpieza de outputs
# ===========================

.PHONY: prep-clean
prep-clean:
	@echo "[clean] Eliminando $(OUT_DIR) y featurizers..."
	@rm -rf "$(OUT_DIR)" "data/prep/textfeats" "data/prep_auto/textfeats"

# ===========================
# Validación de datasets preprocesados
# ===========================

.PHONY: prep-validate
prep-validate:
	@echo "[validate] Validando datasets en $(OUT_DIR)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.validate_prep_dir \
		--dir "$(OUT_DIR)"

# ===========================
# spaCy: instalación y modelo
# ===========================

.PHONY: spacy-install
spacy-install:
	@echo "[spacy] Instalando spaCy y extras..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r backend/spacy-requirements.txt

.PHONY: spacy-model
spacy-model:
	@echo "[spacy] Descargando modelo es_core_news_md..."
	@$(PYTHON) -m spacy download es_core_news_md

# ===========================
# Batch de preprocesamiento + spaCy
# ===========================

# Alias práctico: usa el batch real (cmd_preprocesar_batch)
.PHONY: prep-batch
prep-batch:
	@$(MAKE) prep-all

# ===========================
# Diagnóstico / sandbox para RBM manual
# ===========================

# Test manual de RBM y BM (script real: test_rbm_bm_manual)
.PHONY: test-manual-bm
test-manual-bm:
	@mkdir -p reports
	@echo "[test] Probando RBM/BM manual con dataset data/prep_auto/dataset_ejemplo.parquet..."
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.test_rbm_bm_manual

# Entrenamiento manual de RBM (flujo principal)
# Entrenamiento manual de RBM (flujo principal)
.PHONY: train-rbm-manual
train-rbm-manual:
	@mkdir -p reports
	@echo "[train] RBM manual con PYTHON=$(PYTHON)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_train_rbm_manual \
		--in "data/prep_auto/dataset_ejemplo.parquet" \
		--out-dir "reports" \
		--model "rbm" \
		--n-hidden 64 \
		--lr 0.01 \
		--epochs 10 \
		--batch-size 64 \
		--binarize-input \
		--input-bin-threshold 0.5 \
		--cd-k 1

# ===========================
# Bloque Día 7: limpieza, administración, dev FE/BE, diagnóstico
# ===========================

ENV ?= .env
-include $(ENV)
export

API_HOST ?= 127.0.0.1
API_PORT ?= 8000

NC_RETENTION_DAYS       ?= 90
NC_KEEP_LAST            ?= 3
NC_EXCLUDE_GLOBS        ?=
NC_TRASH_DIR            ?= .trash
NC_TRASH_RETENTION_DAYS ?= 7
NC_ADMIN_TOKEN          ?= dev-admin-token

# Validación de datasets (helpers)
NC_DATASET_ID ?= docentes
NC_SAMPLE_CSV ?= examples/docentes.csv

# CORS / límite de subida
NC_ALLOWED_ORIGINS ?= http://localhost:5173
NC_MAX_UPLOAD_MB   ?= 10

# Rutas comunes para backend y frontend
BACKEND_SRC  ?= backend/src
BACKEND_APP  ?= neurocampus.app.main:app
FRONTEND_DIR ?= frontend

# ----------------------------------------------------------------------------- #
# Ayuda
# ----------------------------------------------------------------------------- #

.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  help                        - Mostrar este mensaje."
	@echo "  clean-inventory             - Ver inventario local de artefactos (herramienta CLI)."
	@echo "  clean-artifacts-dry-run     - Simulación de limpieza local (no borra, muestra plan)."
	@echo "  clean-artifacts             - Limpieza local real (mueve a .trash/)."
	@echo "  admin-inventory             - Inventario remoto vía endpoint /admin/cleanup/inventory."
	@echo "  admin-clean                 - Limpieza remota vía endpoint /admin/cleanup (force)."
	@echo "  be-dev                      - Levantar backend en modo desarrollo (uvicorn)."
	@echo "  be-test                     - Ejecutar pruebas del backend (pytest)."
	@echo "  fe-dev                      - Levantar frontend (Vite) en :5173."
	@echo "  fe-test                     - Ejecutar pruebas del frontend (vitest)."
	@echo "  validate-sample             - Enviar CSV de ejemplo a /datos/validar."
	@echo "  prep-one                    - Preprocesar un dataset de ejemplo (NC_SAMPLE_CSV)."
	@echo "  prep-all                    - Preprocesar todos los CSV de examples/ y synthetic/."
	@echo "  prep-validate               - Validar estructura de los .parquet en $(OUT_DIR)."

# ----------------------------------------------------------------------------- #
# Limpieza de artefactos y cache (local, vía herramienta CLI)
# ----------------------------------------------------------------------------- #

.PHONY: clean-inventory
clean-inventory:
	@$(PYTHON) -m tools.cleanup --inventory

.PHONY: clean-artifacts-dry-run
clean-artifacts-dry-run:
	@$(PYTHON) -m tools.cleanup --dry-run \
		--retention-days $${NC_RETENTION_DAYS:-90} \
		--keep-last $${NC_KEEP_LAST:-3} \
		--exclude-globs "$${NC_EXCLUDE_GLOBS:-}"

.PHONY: clean-artifacts
clean-artifacts:
	@$(PYTHON) -m tools.cleanup --force \
		--retention-days $${NC_RETENTION_DAYS:-90} \
		--keep-last $${NC_KEEP_LAST:-3} \
		--exclude-globs "$${NC_EXCLUDE_GLOBS:-}" \
		--trash-dir "$${NC_TRASH_DIR:-.trash}" \
		--trash-retention-days $${NC_TRASH_RETENTION_DAYS:-7}

# ----------------------------------------------------------------------------- #
# Administración (vía API)
# ----------------------------------------------------------------------------- #

.PHONY: admin-inventory
admin-inventory:
	@curl -s \
		-H "Authorization: Bearer $(NC_ADMIN_TOKEN)" \
		"http://$(API_HOST):$(API_PORT)/admin/cleanup/inventory?retention_days=$(NC_RETENTION_DAYS)&keep_last=$(NC_KEEP_LAST)" | jq .

.PHONY: admin-clean
admin-clean:
	@curl -s -X POST \
		-H "Authorization: Bearer $(NC_ADMIN_TOKEN)" \
		-H "Content-Type: application/json" \
		-d "{\"retention_days\":$(NC_RETENTION_DAYS),\"keep_last\":$(NC_KEEP_LAST),\"trash_dir\":\"$(NC_TRASH_DIR)\",\"trash_retention_days\":$(NC_TRASH_RETENTION_DAYS)}" \
		"http://$(API_HOST):$(API_PORT)/admin/cleanup" | jq .

# ----------------------------------------------------------------------------- #
# Desarrollo backend / frontend
# ----------------------------------------------------------------------------- #

.PHONY: be-dev
be-dev:
	@echo ">> Iniciando uvicorn (desarrollo). Límite de subida via middleware: $${NC_MAX_UPLOAD_MB:-10} MB"
	@uvicorn $(BACKEND_APP) --app-dir $(BACKEND_SRC) \
		--host $${API_HOST} --port $${API_PORT} --reload

.PHONY: be-test
be-test:
	@PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m pytest -q

.PHONY: fe-dev
fe-dev:
	@cd $(FRONTEND_DIR) && npm run dev

.PHONY: fe-test
fe-test:
	@cd $(FRONTEND_DIR) && npm run test:run

# ----------------------------------------------------------------------------- #
# Diagnóstico: validación de datasets vía API
# ----------------------------------------------------------------------------- #

.PHONY: validate-sample
validate-sample:
	@test -f "$(NC_SAMPLE_CSV)" || (echo "ERROR: No existe $(NC_SAMPLE_CSV). Ajusta NC_SAMPLE_CSV o agrega un ejemplo." && exit 1)
	@echo ">> Validando archivo '$(NC_SAMPLE_CSV)' como dataset_id=$(NC_DATASET_ID) contra http://$(API_HOST):$(API_PORT)/datos/validar"
	@curl -s -F "file=@$(NC_SAMPLE_CSV)" -F "dataset_id=$(NC_DATASET_ID)" \
		"http://$(API_HOST):$(API_PORT)/datos/validar" | jq .
