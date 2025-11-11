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

# Auto-detección de columna de texto por defecto
TEXT_COLS ?= auto

.PHONY: which-python
which-python:
	@echo "Using PYTHON=$(PYTHON)"

# ===========================
# Virtualenv/requirements
# ===========================

.PHONY: venv-backend-create
venv-backend-create:
ifeq ($(OS),Windows_NT)
	python -m venv backend/.venv
	./backend/.venv/Scripts/python.exe -m pip install --upgrade pip
else
	python -m venv backend/.venv
	./backend/.venv/bin/python -m pip install --upgrade pip
endif

.PHONY: install-reqs
install-reqs:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r backend/requirements.txt

# ===========================
# Preprocesamiento
# ===========================

# make prep-one IN=examples/dataset_ejemplo.csv OUT=data/prep_auto/dataset_ejemplo.parquet
.PHONY: prep-one
prep-one:
	@if [ -z "$(IN)" ] || [ -z "$(OUT)" ]; then \
		echo "Uso: make prep-one IN=<csv> OUT=<parquet>"; exit 1; \
	fi
	@mkdir -p $(dir $(OUT))
	@echo "[one] Procesando: $(IN) → $(OUT)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
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

.PHONY: prep-all
prep-all:
	@mkdir -p "$(OUT_DIR)"
	@echo "[batch] Buscando CSV en 'examples' y 'examples/synthetic' (si existe)..."
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_batch \
		--in-dirs "examples,$(EXAMPLES)/synthetic" \
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
# Validación (sin heredoc)
# ===========================

.PHONY: prep-validate
prep-validate:
	@echo "[validate] Inspeccionando parquet en $(OUT_DIR)"
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.validate_prep_dir \
		--dir "$(OUT_DIR)" \
		--must-exist-cols "accepted_by_teacher,sentiment_label_teacher,sentiment_conf,has_text" \
		--require-any-prefix "feat_t_"

# ===========================
# spaCy (opcional)
# ===========================

.PHONY: spacy-install
spacy-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install "spacy>=3.7,<4" "spacy-lookups-data>=1.0.5" emoji>=2.12.1

.PHONY: spacy-model
spacy-model: spacy-install
	$(PYTHON) -m spacy download es_core_news_sm

# ===========================
# Utilidades
# ===========================

.PHONY: prep-clean
prep-clean:
	@echo "[clean] Eliminando $(OUT_DIR) y featurizers..."
	@rm -rf "$(OUT_DIR)" "data/prep/textfeats" "data/prep_auto/textfeats"

.PHONY: test-manual-bm
test-manual-bm:
	@PYTHONPATH="$(SRC_DIR)$(PATHSEP)$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.test_rbm_bm_manual