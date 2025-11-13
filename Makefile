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

# Rutas y variables comunes
REPO_ROOT := $(shell pwd)
SRC_DIR   := $(REPO_ROOT)/backend/src
EXAMPLES  := examples
OUT_DIR   := data/prep_auto

# -------- Variables del pipeline (sin espacios basura) --------
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
# Preproceso - Paso 1: texto crudo -> parquet básico
# ===========================

# Preprocesar un solo archivo de ejemplo.
# Usa:
#   BETO_MODE   (simple|full)
#   EXAMPLES    directorio con los CSV de ejemplo (por defecto: examples)
#   OUT_DIR     directorio de salida (por defecto: data/prep_auto)
#
# Ejemplo:
#   make prep-one BETO_MODE=simple
.PHONY: prep-one
prep-one:
	@mkdir -p "$(OUT_DIR)"
	@echo "[prep-one] Usando BETO_MODE=$(BETO_MODE), ejemplos en $(EXAMPLES), salida en $(OUT_DIR)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocess_one \
		--examples "$(EXAMPLES)" \
		--out-dir "$(OUT_DIR)" \
		--beto-mode "$(BETO_MODE)"

# Preprocesar todos los datasets de ejemplo encontrados en EXAMPLES.
.PHONY: prep-all
prep-all:
	@mkdir -p "$(OUT_DIR)"
	@echo "[prep-all] Buscando CSV en 'examples' y 'examples/synthetic' (si existe)..."
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocess_all \
		--examples "$(EXAMPLES)" \
		--out-dir "$(OUT_DIR)" \
		--beto-mode "$(BETO_MODE)"

# ===========================
# Preproceso - Limpieza de outputs
# ===========================

.PHONY: prep-clean
prep-clean:
	@echo "[clean] Eliminando $(OUT_DIR) y featurizers..."
	@rm -rf "$(OUT_DIR)" "data/prep/textfeats" "data/prep_auto/textfeats"

# ===========================
# Validación / split de datasets preprocesados
# ===========================

# Valida un archivo parquet dentro de OUT_DIR:
#   - Chequea columnas mínimas
#   - Aplica splits train/val/test
# Variables:
#   - VAL_SIZE, TEST_SIZE, RANDOM_STATE
.PHONY: prep-validate
prep-validate:
	@echo "[validate] Validando datasets en $(OUT_DIR)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_validate_preprocessed \
		--out-dir "$(OUT_DIR)" \
		--val-size "$(VAL_SIZE)" \
		--test-size "$(TEST_SIZE)" \
		--random-state "$(RANDOM_STATE)"

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

# Preprocesa texto y genera featurizers (spaCy + features numéricas)
.PHONY: prep-batch
prep-batch:
	@mkdir -p "$(OUT_DIR)"
	@echo "[batch] Ejecutando pipeline de preprocesamiento + spaCy"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_prepro_with_spacy \
		--examples "$(EXAMPLES)" \
		--out-dir "$(OUT_DIR)" \
		--beto-mode "$(BETO_MODE)" \
		--min-tokens "$(MIN_TOKENS)" \
		--max-tokens "$(MAX_TOKENS)"

# ===========================
# Diagnóstico / sandbox para RBM manual
# ===========================

# Test manual de RBM con un subconjunto pequeño para debugging.
.PHONY: test-manual-bm
test-manual-bm:
	@mkdir -p reports
	@echo "[test] Probando RBM manual con un subset pequeño..."
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_test_rbm_manual \
		--in "data/prep_auto/dataset_ejemplo.parquet" \
		--out-dir "reports" \
		--sample-size 512 \
		--threshold "$(THRESHOLD)" \
		--margin "$(MARGIN)" \
		--neu-min "$(NEU_MIN)"

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
		--lr 0.05 \
		--epochs 2 \
		--batch-size 64 \
		--binarize-input 1 \
		--input-bin-threshold 0.5

# ===========================
# Bloque Día 7: limpieza, administración, dev FE/BE, diagnóstico
# ===========================

# Variables de entorno para API y limpieza (se pueden sobreescribir o venir de .env)
ENV ?= .env
-include $(ENV)
export

API_HOST ?= 127.0.0.1
API_PORT ?= 8000

# Administración de limpieza (valores por defecto)
NC_RETENTION_DAYS       ?= 90
NC_KEEP_LAST            ?= 3
NC_EXCLUDE_GLOBS        ?=
NC_TRASH_DIR            ?= .trash
NC_TRASH_RETENTION_DAYS ?= 7
NC_ADMIN_TOKEN          ?= dev-admin-token

# Validación de datasets (helpers)
NC_DATASET_ID ?= docentes
NC_SAMPLE_CSV ?= examples/docentes.csv

# CORS / límite de subida (coherente con main.py)
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

# Borrado real (mueve a .trash/). Requiere --force.
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

# Ejecuta limpieza remota en el backend (force=true).
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

# Ejecutar pruebas del backend. Forzamos PYTHONPATH para resolver imports de backend/src.
.PHONY: be-test
be-test:
	@PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m pytest -q

# Frontend: desarrollo y pruebas
.PHONY: fe-dev
fe-dev:
	@cd $(FRONTEND_DIR) && npm run dev

.PHONY: fe-test
fe-test:
	@cd $(FRONTEND_DIR) && npm run test:run

# ----------------------------------------------------------------------------- #
# Diagnóstico: validación de datasets
# ----------------------------------------------------------------------------- #

# Helper: envía un CSV de ejemplo al endpoint /datos/validar del backend.
# Variables:
#   - NC_SAMPLE_CSV (ruta del archivo CSV a enviar)
#   - NC_DATASET_ID (identificador lógico del dataset, por defecto "docentes")
.PHONY: validate-sample
validate-sample:
	@test -f "$(NC_SAMPLE_CSV)" || (echo "ERROR: No existe $(NC_SAMPLE_CSV). Ajusta NC_SAMPLE_CSV o agrega un ejemplo." && exit 1)
	@echo ">> Validando archivo '$(NC_SAMPLE_CSV)' como dataset_id=$(NC_DATASET_ID) contra http://$(API_HOST):$(API_PORT)/datos/validar"
	@curl -s -F "file=@$(NC_SAMPLE_CSV)" -F "dataset_id=$(NC_DATASET_ID)" \
		"http://$(API_HOST):$(API_PORT)/datos/validar" | jq .
