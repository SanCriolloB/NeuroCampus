# NeuroCampus
Bootstrap del repositorio (Día 1).

## Cómo correr
### Backend (FastAPI)
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn neurocampus.app.main:app --reload --app-dir backend/src

### Frontend (Vite + React + TS)
cd frontend
npm install
npm run dev

## Flujo de ramas
main (protegida), develop (protegida), ramas de feature desde develop.
