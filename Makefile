# ===========================
# Makefile - NeuroCampus (simple y robusto)
# ===========================

# -------- Detección robusta de PYTHON --------
WIN_ROOT    := ./.venv/Scripts/python.exe
WIN_BACKEND := ../.venv/Scripts/python.exe
NIX_ROOT    := .venv/bin/python
NIX_BACKEND := ../.venv/bin/python

ifeq ($(OS),Windows_NT)
  # Si existe la venv estilo Windows (Scripts/), úsala
  ifneq ("$(wildcard $(WIN_ROOT))","")
    PY_ROOT    := $(WIN_ROOT)
    PY_BACKEND := $(WIN_BACKEND)
  else
    # Fallback: estilo POSIX (bin/) aunque estemos en Windows (MSYS, etc.)
    PY_ROOT    := $(NIX_ROOT)
    PY_BACKEND := $(NIX_BACKEND)
  endif
  PATHSEP := ;
else
  PY_ROOT    := $(NIX_ROOT)
  PY_BACKEND := $(NIX_BACKEND)
  PATHSEP    := :
endif

PYTHON ?= $(PY_ROOT)


# ===========================
# Variables principales
# ===========================

SRC_DIR       := backend/src
BACKEND_DIR   := backend
FRONTEND_DIR  := frontend
REPORTS_DIR   := reports
DATA_DIR      := data
EXAMPLES_DIR  := examples
OUT_DIR      ?= $(DATA_DIR)/prep_auto

# -------- Variables del pipeline (sin espacios basura) --------
BETO_MODE       ?= simple
MIN_TOKENS      ?= 1
KEEP_EMPTY_TEXT ?= 1
TEXT_FEATS      ?= tfidf_lsa
TFIDF_MIN_DF    ?= 1
TFIDF_MAX_DF    ?= 1
BETO_MODEL      ?= finiteautomata/beto-sentiment-analysis
BATCH_SIZE      ?= 32
THRESHOLD       ?= 0.45
MARGIN          ?= 0.05
NEU_MIN         ?= 0.10

# ===========================
# Targets por defecto
# ===========================

.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  venv             - Crear entorno virtual en raiz y backend"
	@echo "  install          - Instalar dependencias (backend + frontend)"
	@echo "  be-install       - Instalar dependencias backend"
	@echo "  fe-install       - Instalar dependencias frontend"
	@echo "  be-dev           - Correr backend en modo desarrollo"
	@echo "  fe-dev           - Correr frontend en modo desarrollo"
	@echo "  lint             - Ejecutar linters en backend"
	@echo "  be-deps-check     - Verificar coherencia de dependencias (pip check)"
	@echo "  be-ci            - Ejecutar gates backend: lint + pip check + tests"
	@echo "  prep-one         - Preprocesar un solo CSV"
	@echo "  prep-all         - Preprocesar todos los CSV de examples/"
	@echo "  test-manual-bm   - Probar RBM/BM manual"
	@echo "  train-rbm-manual - Entrenar RBM manual en dataset_ejemplo"
	@echo "  train-dbm-manual - Entrenar DBM manual en dataset_ejemplo"
	@echo "  rbm-audit        - Auditoria k-fold de modelos RBM/BM"
	@echo "  rbm-search       - Busqueda de hiperparametros RBM/BM"
	@echo "  be-test          - Tests backend"
	@echo "  fe-test          - Tests frontend"
	@echo "  validate-sample  - Validar dataset de ejemplo via API"
	@echo "  docs-html  	  - Crear documentacion"
# ===========================
# Entorno virtual
# ===========================

.PHONY: venv
venv:
	@echo ">> Para crear el entorno virtual en la raiz y activar, copiar y pegar lo siguiente en la terminal:"
	@echo "# Para crear el entorno virtual en raiz:"
	@echo "   python -m venv .venv"
	@echo "# Para activar en Git Bash (raiz):"
	@echo "   source .venv/Scripts/activate"
	
# ===========================
# Instalación de dependencias
# ===========================

.PHONY: install
install: be-install fe-install

.PHONY: be-install
be-install:
	@echo ">> Instalando dependencias backend con PY_BACKEND=$(PY_BACKEND)"
	cd backend && $(PY_BACKEND) -m pip install --upgrade pip
	cd backend && $(PY_BACKEND) -m pip install -r requirements.txt
	cd backend && $(PY_BACKEND) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install "spacy>=3.7,<4" "spacy-lookups-data>=1.0.5" emoji>=2.12.1
	$(PYTHON) -m spacy download es_core_news_sm

.PHONY: fe-install
fe-install:
	@echo ">> Instalando dependencias frontend"
	cd $(FRONTEND_DIR) && npm install && npm install recharts && npm install lucide-react

# ===========================
# Backend: desarrollo y tests
# ===========================

.PHONY: be-dev
be-dev:
	@echo ">> Ejecutando backend en modo desarrollo con PY_BACKEND=$(PY_BACKEND)"
	cd backend && PYTHONPATH="src$(PATHSEP)$$PYTHONPATH" \
	$(PY_BACKEND) -m uvicorn neurocampus.app.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: be-deps-check
be-deps-check:
	@echo ">> Verificando dependencias instaladas (pip check) en backend"
	cd backend && $(PY_BACKEND) -m pip check

.PHONY: be-ci
be-ci: lint be-deps-check be-test
	@echo ">> Gates backend OK (lint + pip check + pytest)"

.PHONY: be-test
be-test:
	@echo ">> Ejecutando tests backend con PYTHON=$(PYTHON)"
	@PYTHONPATH="backend/src$(PATHSEP).$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m pytest -q tests


.PHONY: lint
lint:
	@echo ">> Ejecutando linters (ruff + mypy) en backend"
	cd $(BACKEND_DIR) && PYTHONPATH="src$(PATHSEP)$$PYTHONPATH" \
	$(PY_BACKEND) -m ruff check .
	cd $(BACKEND_DIR) && PYTHONPATH="src$(PATHSEP)$$PYTHONPATH" \
	$(PY_BACKEND) -m mypy src/neurocampus

# ===========================
# Frontend: desarrollo y tests
# ===========================

.PHONY: fe-dev
fe-dev:
	@cd $(FRONTEND_DIR) && npm run dev

.PHONY: fe-test
fe-test:
	@cd $(FRONTEND_DIR) && npm run test:run

# ===========================
# Preprocesamiento de datos
# ===========================

# Preprocesar un solo CSV usando el pipeline real (cmd_preprocess_one)
.PHONY: prep-one
prep-one:
	@if [ -z "$(IN)" ] || [ -z "$(OUT)" ]; then \
		echo "Uso: make prep-one IN=<csv> OUT=<parquet>"; exit 1; \
	fi
	@mkdir -p $(dir $(OUT))
	@echo "[one] Procesando: $(IN) → $(OUT)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP).$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_beto \
		--in "$(IN)" \
		--out "$(OUT)" \
		--beto-mode "$(strip $(BETO_MODE))" \
		--min-tokens "$(strip $(MIN_TOKENS))" \
		--text-feats "$(strip $(TEXT_FEATS))" \
		--beto-model "$(strip $(BETO_MODEL))" \
		--batch-size "$(strip $(BATCH_SIZE))" \
		--threshold "$(strip $(THRESHOLD))" \
		--margin "$(strip $(MARGIN))" \
		--neu-min "$(strip $(NEU_MIN))" \
		--tfidf-min-df "$(strip $(TFIDF_MIN_DF))" \
		--tfidf-max-df "$(strip $(TFIDF_MAX_DF))" \
		$(if $(filter $(KEEP_EMPTY_TEXT),1),--keep-empty-text,) \
		$(if $(filter-out auto,$(TEXT_COLS)),--text-col "$(strip $(TEXT_COLS))",)

# Preprocesar todos los CSV de examples/ y examples/synthetic/ (si existe)
.PHONY: prep-all
prep-all:
	@mkdir -p "$(OUT_DIR)"
	@echo "[batch] Buscando CSV en '$(EXAMPLES_DIR)' y '$(EXAMPLES_DIR)/synthetic' (si existe)..."
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP).$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_batch \
		--in-dirs "$(EXAMPLES_DIR),$(EXAMPLES_DIR)/synthetic" \
		--out-dir "$(OUT_DIR)" \
		--text-cols "$(strip $(TEXT_COLS))" \
		--beto-mode "$(strip $(BETO_MODE))" \
		--min-tokens "$(strip $(MIN_TOKENS))" \
		--keep-empty-text \
		--tfidf-min-df "$(strip $(TFIDF_MIN_DF))" \
		--tfidf-max-df "$(strip $(TFIDF_MAX_DF))" \
		--text-feats "$(strip $(TEXT_FEATS))" \
		--beto-model "$(strip $(BETO_MODEL))" \
		--batch-size "$(strip $(BATCH_SIZE))" \
		--threshold "$(strip $(THRESHOLD))" \
		--margin "$(strip $(MARGIN))" \
		--neu-min "$(strip $(NEU_MIN))"
		

# ===========================
# RBM/BM manual (bloque de pruebas)
# ===========================

# Test sencillo para RBM y BM (script real: test_rbm_bm_manual)
.PHONY: test-manual-bm
test-manual-bm:
	@mkdir -p reports
	@echo "[test] Probando RBM/BM manual con dataset data/prep_auto/dataset_ejemplo.parquet..."
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.test_rbm_bm_manual

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

# Entrenamiento manual de DBM (flujo principal)
.PHONY: train-dbm-manual
train-dbm-manual:
	@mkdir -p reports/dbm_manual
	@echo "[train] DBM manual con PYTHON=$(PYTHON)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_train_dbm_manual \
		--in "data/prep_auto/dataset_ejemplo.parquet" \
		--out-dir "reports/dbm_manual" \
		--n-hidden1 64 \
		--n-hidden2 32 \
		--lr 0.01 \
		--cd-k 1 \
		--epochs 10 \
		--batch-size 64

# Auditoría k-fold de RBM/BM
.PHONY: rbm-audit
rbm-audit:
	@mkdir -p artifacts/runs
	@echo "[audit] Ejecutando auditoría k-fold RBM/BM con configs/rbm_audit.yaml"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.models.audit_kfold \
		--config "configs/rbm_audit.yaml"

# Búsqueda de hiperparámetros RBM/BM
.PHONY: rbm-search
rbm-search:
	@mkdir -p artifacts/runs
	@echo "[search] Ejecutando búsqueda de hiperparámetros RBM/BM con configs/rbm_search.yaml"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.models.hparam_search \
		--config "configs/rbm_search.yaml"


# ===========================
# Bloque: Limpieza, administración, dev FE/BE, diagnóstico
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
NC_ADMIN_TOKEN          ?=

NC_SAMPLE_CSV ?= ./examples/review_sample.csv
NC_DATASET_ID ?= demo-reviews

# ----------------------------------------------------------------------------- #
# Diagnóstico: validación de datasets vía API
# ----------------------------------------------------------------------------- #

.PHONY: validate-sample
validate-sample:
	@test -f "$(NC_SAMPLE_CSV)" || (echo "ERROR: No existe $(NC_SAMPLE_CSV). Ajusta NC_SAMPLE_CSV o agrega un ejemplo." && exit 1)
	@echo ">> Validando archivo '$(NC_SAMPLE_CSV)' como dataset_id=$(NC_DATASET_ID) contra http://$(API_HOST):$(API_PORT)/datos/validar"
	@curl -s -F "file=@$(NC_SAMPLE_CSV)" -F "dataset_id=$(NC_DATASET_ID)" \
		"http://$(API_HOST):$(API_PORT)/datos/validar" | jq .

# HTML de la documentacion
.PHONY: docs-html
docs-html:
	@cd docs && make html