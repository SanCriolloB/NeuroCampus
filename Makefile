# ============================
# NeuroCampus — Makefile total
# ============================
# Requisitos:
#   - Python accesible como `python` (o ajusta PYTHON).
#   - Estructura de proyecto con layout "src" en backend/src.
#
# Flujo end-to-end:
#   1) Preprocesamiento por archivo (prep-one) o masivo (prep-all / prep-all-probs)
#   2) Validación de .parquet (prep-validate)
#   3) Entrenamiento RBM pura (train-rbm-pura)
#   4) Datos sintéticos (synth-gen + synth-prep)
#   5) Tests mínimos (tests)
#   6) Auditoría reproducible (audit)
#   7) Limpieza (clean-*)
#
# Variables ajustables en línea, p.ej.:
#   make prep-all BETO_MODE=probs TFIDF_MIN_DF=1 TFIDF_MAX_DF=1.0

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# ---- Python / Rutas ----
PYTHON ?= python
PROJ_ROOT := $(PWD)
SRC_DIR   := $(PROJ_ROOT)/backend/src
export PYTHONPATH := $(SRC_DIR)

# ---- Dirs de trabajo ----
PREP_DIR        := data/prep_auto
PREP_TEXTFEATS  := $(PREP_DIR)/textfeats
RBM_RUNS        := artifacts/runs/rbm_pura
SYNTH_EXAMPLES  := examples/synthetic

# ---- Parámetros NLP por defecto (overrideables) ----
TEXT_COLS     ?= comentario,observaciones
BETO_MODE     ?= simple              # simple | probs
MIN_TOKENS    ?= 1
KEEP_EMPTY    ?= 1                   # 1: --keep-empty-text, 0: no
TFIDF_MIN_DF  ?= 1.0
TFIDF_MAX_DF  ?= 1.0
TEXT_FEATS    ?= tfidf_lsa           # none | tfidf_lsa
BETO_MODEL    ?= finiteautomata/beto-sentiment-analysis
BATCH_SIZE    ?= 32
THRESHOLD     ?= 0.45
MARGIN        ?= 0.05
NEU_MIN       ?= 0.10

# ---- Parámetros de entrenamiento RBM ----
N_HIDDEN     ?= 64
BATCH_TRAIN  ?= 64
CD_K         ?= 1
EPOCHS       ?= 5

# ---- Utilidad: flag condicional keep-empty-text ----
ifeq ($(KEEP_EMPTY),0)
KEEP_FLAG :=
else
KEEP_FLAG := --keep-empty-text
endif

# ---- Utilidad: si OUT no se pasa en prep-one, derivarlo de IN ----
# OUT = data/prep_auto/<nombre>.parquet
OUT ?= $(PREP_DIR)/$(notdir $(basename $(IN))).parquet

# ----------------
#      Targets
# ----------------

.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  make prep-one IN=examples/Evaluacion.csv [OUT=...]   -> Preprocesa un CSV a .parquet"
	@echo "  make prep-all                                        -> Preprocesa en masa examples/ y examples/synthetic/"
	@echo "  make prep-all-probs                                   -> Igual que prep-all pero BETO_MODE=probs"
	@echo "  make prep-validate                                   -> Valida que .parquet tengan feat_t_* y etiqueta"
	@echo "  make train-rbm-pura                                  -> Entrena RBM pura sobre data/prep_auto/*.parquet"
	@echo "  make synth-gen [ROWS=100000]                         -> Genera CSV sintético (examples/synthetic)"
	@echo "  make synth-prep                                      -> Preprocesa todo lo sintético (usa prep-all)"
	@echo "  make tests                                           -> Corre pruebas clave"
	@echo "  make audit                                           -> Pipeline reproducible con logs y artefactos"
	@echo "  make all                                             -> prep-all + validate + train"
	@echo "  make clean-prep | clean-artifacts | clean-all        -> Limpiezas"
	@echo ""
	@echo "Variables útiles (overrideables):"
	@echo "  TEXT_COLS='comentario,observaciones' BETO_MODE=simple|probs MIN_TOKENS=1 KEEP_EMPTY=1"
	@echo "  TFIDF_MIN_DF=1.0 TFIDF_MAX_DF=1.0 TEXT_FEATS=tfidf_lsa BETO_MODEL='finiteautomata/beto-sentiment-analysis'"
	@echo "  THRESHOLD=0.45 MARGIN=0.05 NEU_MIN=0.10"
	@echo "  N_HIDDEN=64 BATCH_TRAIN=64 CD_K=1 EPOCHS=5"

# -------- Preprocesamiento (un archivo) --------
.PHONY: prep-one
prep-one:
	@mkdir -p "$(PREP_DIR)" "$(PREP_TEXTFEATS)"
ifeq ($(strip $(IN)),)
	$(error Debes pasar IN=/ruta/al.csv)
endif
	@echo "[prep-one] IN=$(IN)"
	@echo "[prep-one] OUT=$(OUT)"
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_beto \
		--in "$(IN)" \
		--out "$(OUT)" \
		--text-col "$(TEXT_COLS)" \
		--beto-mode $(BETO_MODE) \
		--min-tokens $(MIN_TOKENS) $(KEEP_FLAG) \
		--text-feats $(TEXT_FEATS) \
		--text-feats-out-dir "$(PREP_TEXTFEATS)" \
		--beto-model "$(BETO_MODEL)" \
		--batch-size $(BATCH_SIZE) \
		--threshold $(THRESHOLD) \
		--margin $(MARGIN) \
		--neu-min $(NEU_MIN) \
		--tfidf-min-df $(TFIDF_MIN_DF) \
		--tfidf-max-df $(TFIDF_MAX_DF)

# -------- Preprocesamiento (masivo) --------
.PHONY: prep-all
prep-all:
	@mkdir -p "$(PREP_DIR)" "$(PREP_TEXTFEATS)"
	$(PYTHON) -m neurocampus.app.jobs.cmd_preprocesar_batch \
		--in-dirs "examples,$(SYNTH_EXAMPLES)" \
		--out-dir "$(PREP_DIR)" \
		--text-cols "$(TEXT_COLS)" \
		--beto-mode $(BETO_MODE) \
		--min-tokens $(MIN_TOKENS) \
		--tfidf-min-df $(TFIDF_MIN_DF) \
		--tfidf-max-df $(TFIDF_MAX_DF)

# -------- Preprocesamiento (masivo, modo probabilístico) --------
.PHONY: prep-all-probs
prep-all-probs:
	@mkdir -p "$(PREP_DIR)" "$(PREP_TEXTFEATS)"
	$(MAKE) prep-all BETO_MODE=probs

# -------- Validación de .parquet --------
.PHONY: prep-validate
prep-validate:
	@echo "[validate] Revisando feat_t_* y etiqueta en $(PREP_DIR)/*.parquet ..."
	$(PYTHON) - << 'PY'
import glob, pyarrow.parquet as pq
ok=True
files=sorted(glob.glob("$(PREP_DIR)/*.parquet"))
if not files:
    print("[validate] No hay archivos .parquet en $(PREP_DIR)")
    raise SystemExit(1)
for p in files:
    t = pq.read_table(p); cols = t.column_names
    feats = any(c.startswith("feat_t_") for c in cols)
    label = any(c in cols for c in ("y_sent","sentimiento","label_sent"))
    print(p, "feat_t_*:", feats, "label:", label, "n_cols:", len(cols))
    ok = ok and feats and label
raise SystemExit(0 if ok else 1)
PY

# -------- Entrenamiento RBM pura --------
.PHONY: train-rbm-pura
train-rbm-pura:
	@mkdir -p "$(RBM_RUNS)"
	@echo "[train] Entrenando RBM pura sobre $(PREP_DIR)/*.parquet ..."
	set -e
	shopt -s nullglob
	parquets=($(PREP_DIR)/*.parquet)
	if [ $${#parquets[@]} -eq 0 ]; then
		echo "[train] No hay .parquet en $(PREP_DIR). Ejecuta 'make prep-all' primero."
		exit 1
	fi
	for f in "$${parquets[@]}"; do
		echo " → $$f"
		$(PYTHON) backend/scripts/train_rbm_pura.py \
			--data "$$f" \
			--artifacts_dir "$(RBM_RUNS)" \
			--n_hidden $(N_HIDDEN) \
			--batch_size $(BATCH_TRAIN) \
			--cd_k $(CD_K) \
			--epochs $(EPOCHS)
	done
	@ls -lh "$(RBM_RUNS)" || true

# -------- Datos sintéticos --------
.PHONY: synth-gen
synth-gen:
	@mkdir -p "$(SYNTH_EXAMPLES)"
	$(PYTHON) tools/sim/generate_synthetic.py --rows $(or $(ROWS),100000) --out $(SYNTH_EXAMPLES)/synth_evaluaciones.csv

.PHONY: synth-prep
synth-prep: synth-gen
	$(MAKE) prep-all

# -------- Tests mínimos --------
.PHONY: tests
tests:
	pytest -q tests/unit/test_rbm_pura_api.py
	pytest -q tests/api/test_datos_upload_integration.py

# -------- Auditoría reproducible --------
.PHONY: audit
audit:
	@ts="$$(date +'%Y-%m-%d_%H-%M-%S')"
	@AUD="artifacts/audits/$$ts"
	@mkdir -p "$$AUD"
	@echo "[audit] Carpeta: $$AUD"

	@echo "[audit] Paso 1: prep-all (modo estándar)"
	@$(MAKE) prep-all | tee "$$AUD/prep-all.log"

	@echo "[audit] Paso 2: validate"
	@$(MAKE) prep-validate | tee "$$AUD/validate.log"

	@echo "[audit] Paso 3: entrenamiento (parámetros actuales)"
	@$(MAKE) train-rbm-pura | tee "$$AUD/train.log"

	@echo "[audit] Paso 4: tests"
	@$(MAKE) tests | tee "$$AUD/tests.log"

	@echo "[audit] Listado de .parquet usados" > "$$AUD/parquets.txt"
	@ls -1 $(PREP_DIR)/*.parquet >> "$$AUD/parquets.txt" || true

	@echo "[audit] Finalizado. Ver: $$AUD"

# -------- Pipeline completo --------
.PHONY: all
all: prep-all prep-validate train-rbm-pura
	@echo "[all] Preprocesado, validado y entrenado correctamente."

# -------- Limpieza --------
.PHONY: clean-prep
clean-prep:
	rm -rf "$(PREP_DIR)"

.PHONY: clean-artifacts
clean-artifacts:
	rm -rf "artifacts/runs" "artifacts/checkpoints" "artifacts/audits"

.PHONY: clean-all
clean-all: clean-prep clean-artifacts
	@echo "[clean-all] Limpieza de prep y artifacts completada."
