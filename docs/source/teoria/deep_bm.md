# Deep Boltzmann Machines (DBM)

## Introducción

Las **Deep Boltzmann Machines** (DBM) extienden la idea de las RBM y BM a
**múltiples capas ocultas**, formando modelos de energía profundos. El objetivo
es capturar representaciones jerárquicas de los datos, donde capas superiores
aprenden características progresivamente más abstractas.

En términos conceptuales, una DBM se parece a una pila de RBM con conexiones
entre capas adyacentes, pero con una interpretación conjunta como modelo
probabilístico global.

En NeuroCampus, las DBM se consideran una alternativa avanzada dentro de la
familia de modelos de Boltzmann para explorar representaciones más profundas
de las evaluaciones docentes.

---

## Estructura del modelo

Una DBM típica puede representarse como:

- Capa visible \(v\).
- Varias capas ocultas: \(h^{(1)}, h^{(2)}, \ldots, h^{(L)}\).

Las conexiones se dan únicamente entre capas **adyacentes**:

- Pesos \(W^{(0)}\) entre \(v\) y \(h^{(1)}\),
- Pesos \(W^{(1)}\) entre \(h^{(1)}\) y \(h^{(2)}\), y así sucesivamente.

La energía de un estado completo se escribe como una suma de interacciones
entre capas vecinas y términos de sesgo. De forma esquemática:

\[
E(v, h^{(1)}, \ldots, h^{(L)}) =
- v^T W^{(0)} h^{(1)}
- (h^{(1)})^T W^{(1)} h^{(2)} - \cdots
- \text{sesgos}.
\]

---

## Inferencia aproximada

A diferencia de las RBM, donde las condicionales \(P(h \mid v)\) y
\(P(v \mid h)\) factorizan cómodamente, en la DBM la inferencia es más
compleja:

- Las unidades de una misma capa no son condicionalmente independientes dado
  solo las visibles.
- Para inferir \(h^{(l)}\) se debe tener en cuenta información de las capas
  vecinas \(h^{(l-1)}\) y \(h^{(l+1)}\).

Por ello, se suelen usar:

- Métodos de inferencia variacional,
- aproximaciones mean-field,
- o cadenas de Gibbs más elaboradas.

El entrenamiento conjunto de todas las capas requiere aproximar las
expectativas necesarias para el gradiente, igual que en BM/RBM, pero con más
capas y variables.

---

## Entrenamiento en dos etapas (pre-entrenamiento + afinado)

Una estrategia habitual para entrenar DBM es:

1. **Pre-entrenamiento por capas** (layer-wise pretraining):
   - Entrenar una RBM entre \(v\) y \(h^{(1)}\).
   - Luego una RBM entre \(h^{(1)}\) y \(h^{(2)}\), y así sucesivamente.
   - Esto inicializa los pesos de la DBM con valores razonables.

2. **Entrenamiento conjunto (fine-tuning)**:
   - Tras el pre-entrenamiento, se trata el modelo completo como una DBM.
   - Se aplican algoritmos de entrenamiento aproximado (variacional, CD, PCD,
     etc.) para ajustar conjuntamente todas las capas.

Este enfoque se inspira en la idea de que cada RBM intermedia aprende una
representación útil, que luego se refina cuando se consideran todas las capas
al mismo tiempo.

---

## Ventajas y desafíos

### Ventajas

- Capacidad de representar estructuras complejas y jerárquicas:
  - capas bajas capturan patrones locales,
  - capas altas capturan patrones más globales.
- Puede lograr mejores representaciones que una sola RBM, especialmente en
  problemas con estructura de alto nivel.

### Desafíos

- Entrenamiento significativamente más costoso que RBM o BM simples.
- Requiere técnicas más sofisticadas de inferencia aproximada.
- Mayor sensibilidad a la inicialización y a la configuración de
  hiperparámetros.

Por estos motivos, en contextos prácticos se suele comparar el esfuerzo y
beneficio de usar DBM frente a modelos más sencillos.

---

## Papel de las DBM en NeuroCampus

En NeuroCampus, la DBM se plantea como:

- Una **extensión profunda** de la RBM manual.
- Un experimento avanzado para capturar patrones de alto nivel en:
  - evaluaciones históricas,
  - perfiles de docentes,
  - contextos de asignaturas y programas.

La implementación se encuentra en el módulo `neurocampus.models.dbm_manual`,
que se apoya en componentes de RBM manual para la definición de capas y en
estrategias específicas para su entrenamiento. El objetivo es:

- Comparar el rendimiento de la DBM con RBM y BM.
- Analizar si las representaciones profundas aportan una mejora tangible en la
predicción de rendimiento docente y en los indicadores que se muestran en
Predicciones y Dashboard.
