# Arquitectura — NeuroCampus (Borrador Día 1)
<!--
Objetivo del día 1:
- Dejar clara la arquitectura a alto nivel.
- Fijar patrones y límites de contexto.
- Dar un mapa mínimo para que backend/frontend y data se coordinen.
-->

## 1. Visión general
- **Dominio**: análisis educativo (datasets, modelos, predicción, jobs).
- **Estilo**: servicios modulares (FastAPI) + SPA (React/Vite).
- **Tronco común**: contratos HTTP con JSON (+ TypeScript types para FE).

## 2. Capas y módulos
- **Data** (ingesta/validación): adapters a almacenamiento, validaciones.
- **Models** (entrenamiento/publicación): ciclo de vida del modelo.
- **Prediction** (online/batch): entradas limpias → salidas con postproceso.
- **Jobs** (orquestación/estado): seguimiento de tareas asíncronas.

## 3. Patrones (acordados)
- **Strategy**, **Template Method**, **Facade**, **Chain of Responsibility**,
  **Observer**, **Command**, **Adapter**.
<!-- Nota: se detallarán con ejemplos en días 2–6. -->

## 4. Diagramas (borradores)
### 4.1 Componentes (mermaid)
```mermaid
flowchart LR
  UI[React SPA] -->|REST JSON| API[FastAPI]
  subgraph FastAPI
    D[Data] --- M[Models] --- P[Prediction] --- J[Jobs]
  end
  D -->|valida| DS[(Storage)]
  M -->|snapshot| REG[(Model Registry)]
  P -->|lee modelos| REG
  J -->|publica eventos| BUS[(Event Bus)]