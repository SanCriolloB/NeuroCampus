# ===========================
# Makefile - NeuroCampus
# Detección robusta de PYTHON + jobs de preprocessing
# ===========================

SHELL := /usr/bin/env bash

# -------- Detección robusta de PYTHON --------
# Prioridad: backend/.venv -> ./.venv -> python del sistema
ifeq ($(OS),Windows_NT)
PY_BACKEND := ./backend/.venv/Scripts/python.exe
PY_ROOT    := ./.venv/Scripts/python.exe
else
PY_BACKEND := ./backend/.venv/bin/python
PY_ROOT    := ./.venv/bin/python
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

# Variables por defecto para el pipeline de texto
BETO_MODE        ?= simple          # puedes poner 'probs' si lo prefieres por defecto
MIN_TOKENS       ?= 1
KEEP_EMPTY_TEXT  ?= 1               # 1 → añade --keep-empty-text
TEXT_FEATS       ?= tfidf_lsa
TFIDF_MIN_DF     ?= 1.0
TFIDF_MAX_DF     ?= 1.0
BETO_MODEL       ?= finiteautomata/beto-sentiment-analysis
BATCH_SIZE       ?= 32
THRESHOLD        ?= 0.45
MARGIN           ?= 0.05
NEU_MIN          ?= 0.10

# Auto-detección de columna de texto: por defecto no forzamos --text-col
TEXT_COLS ?= auto

# Utilidad: imprime el intérprete elegido
.PHONY: which-python
which-python:
	@echo "Using PYTHON=$(PYTHON)"

# ===========================
# Virtualenvs de conveniencia
# ===========================

# Crea backend/.venv (si no existe) con pip actualizado
.PHONY: venv-backend-create
venv-backend-create:
ifeq ($(OS),Windows_NT)
	python -m venv backend/.venv
	./backend/.venv/Scripts/python.exe -m pip install --upgrade pip
else
	python -m venv backend/.venv
	./backend/.venv/bin/python -m pip install --upgrade pip
endif

# Instala requirements en backend/.venv (usa $(PYTHON) detectado)
.PHONY: install-reqs
install-reqs:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r backend/requirements.txt

# ===========================
# Jobs de preprocesamiento
# ===========================

# Prepara UN archivo CSV a parquet usando el job BETO
# Uso:
#   make prep-one IN=examples/dataset_ejemplo.csv OUT=data/prep_auto/dataset_ejemplo.parquet
.PHONY: prep-one
prep-one:
	@if [ -z "$(IN)" ] || [ -z "$(OUT)" ]; then \
		echo "Uso: make prep-one IN=<csv> OUT=<parquet>"; exit 1; \
	fi
	@mkdir -p $(dir $(OUT))
	@echo "[one] Procesando: $(IN) → $(OUT)"
	@PYTHONPATH="$(SRC_DIR):$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_beto \
		--in "$(IN)" \
		--out "$(OUT)" \
		--beto-mode "$(BETO_MODE)" \
		--min-tokens "$(MIN_TOKENS)" \
		--text-feats "$(TEXT_FEATS)" \
		--beto-model "$(BETO_MODEL)" \
		--batch-size "$(BATCH_SIZE)" \
		--threshold "$(THRESHOLD)" \
		--margin "$(MARGIN)" \
		--neu-min "$(NEU_MIN)" \
		--tfidf-min-df "$(TFIDF_MIN_DF)" \
		--tfidf-max-df "$(TFIDF_MAX_DF)" \
		$(if $(filter $(KEEP_EMPTY_TEXT),1),--keep-empty-text,) \
		$(if $(filter-out auto,$(TEXT_COLS)),--text-col "$(TEXT_COLS)",)

# Prepara TODOS los CSV dentro de examples/ (y examples/synthetic si existe)
.PHONY: prep-all
prep-all:
	@mkdir -p "$(OUT_DIR)"
	@echo "[batch] Buscando CSV en 'examples' y 'examples/synthetic' (si existe)..."
	@PYTHONPATH="$(SRC_DIR):$$PYTHONPATH" \
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_batch \
		--in-dirs "examples,$(EXAMPLES)/synthetic" \
		--out-dir "$(OUT_DIR)" \
		--text-cols "$(TEXT_COLS)" \
		--beto-mode "$(BETO_MODE)" \
		--min-tokens "$(MIN_TOKENS)" \
		--keep-empty-text \
		--tfidf-min-df "$(TFIDF_MIN_DF)" \
		--tfidf-max-df "$(TFIDF_MAX_DF)" \
		--text-feats "$(TEXT_FEATS)" \
		--beto-model "$(BETO_MODEL)" \
		--batch-size "$(BATCH_SIZE)" \
		--threshold "$(THRESHOLD)" \
		--margin "$(MARGIN)" \
		--neu-min "$(NEU_MIN)"

# Validación rápida: imprime esquema y conteos de los parquet generados
.PHONY: prep-validate
prep-validate:
	@echo "[validate] Inspeccionando parquet en $(OUT_DIR)"
	@$(PYTHON) - <<'PY'
import os, sys, json
import pandas as pd
out_dir = "$(OUT_DIR)"
if not os.path.isdir(out_dir):
    print("[validate] No existe el directorio:", out_dir)
    sys.exit(0)
files = [f for f in os.listdir(out_dir) if f.endswith(".parquet")]
if not files:
    print("[validate] No hay archivos .parquet en", out_dir)
    sys.exit(0)

def short(cols):
    cols = list(cols)
    return cols[:10] + (["..."] if len(cols) > 10 else [])

for f in sorted(files):
    p = os.path.join(out_dir, f)
    try:
        df = pd.read_parquet(p)
        print("\n[OK]", f, "→", len(df), "filas")
        print("  columnas:", short(df.columns))
        nn = df.notna().sum().to_dict()
        # imprime algunas métricas clave si existen
        for k in ("p_neg","p_neu","p_pos","sentiment_label_teacher","accepted_by_teacher","feat_t_1"):
            if k in df.columns:
                print(f"  {k}: not-null={nn.get(k,0)}")
    except Exception as e:
        print("\n[ERR]", f, "→", e)
PY

# ===========================
# spaCy opcional (librería + modelo)
# ===========================

.PHONY: spacy-install
spacy-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install "spacy>=3.7,<4" "spacy-lookups-data>=1.0.5" emoji>=2.12.1

.PHONY: spacy-model
spacy-model: spacy-install
	$(PYTHON) -m spacy download es_core_news_sm

# ===========================
# Utilidades extra
# ===========================

# Limpia artefactos de preparación
.PHONY: prep-clean
prep-clean:
	@echo "[clean] Eliminando $(OUT_DIR) y featurizers..."
	@rm -rf "$(OUT_DIR)" "data/prep/textfeats" "data/prep_auto/textfeats"
