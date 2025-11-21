# Máquinas de Boltzmann (BM)

## Introducción

Las **Máquinas de Boltzmann** (BM) son modelos de energía probabilísticos que
definen una distribución sobre un conjunto de variables binarias (o continuas)
usando una función de energía. Se inspiran en sistemas físicos, donde los
estados más probables son aquellos con menor energía.

En NeuroCampus se usan como parte de la familia de modelos para capturar
relaciones complejas entre características derivadas de las evaluaciones
docentes (respuestas numéricas, indicadores agregados, características de
texto, etc.).

---

## Estructura del modelo

Una BM estándar está compuesta por:

- Un conjunto de **unidades visibles** \(v\), que representan los datos de
  entrada.
- Un conjunto de **unidades ocultas** \(h\), que capturan dependencias
  latentes.
- Pesos simétricos \(W\) entre unidades (pueden existir conexiones visibles–
  visibles, ocultas–ocultas y visibles–ocultas).
- Sesgos \(b\) para unidades visibles y \(c\) para unidades ocultas.

No hay una restricción bipartita estricta como en las RBM: pueden existir
conexiones arbitrarias entre nodos.

---

## Energía y distribución de probabilidad

La energía de un estado \((v, h)\) se define típicamente como:

\[
E(v, h) = - v^T W h - b^T v - c^T h + \text{(términos adicionales si hay enlaces vis–vis u oc–oc)}
\]

La probabilidad conjunta sobre todos los estados se define como:

\[
P(v, h) = \frac{1}{Z} \exp\big(-E(v, h)\big),
\]

donde \(Z\) es la **función de partición**:

\[
Z = \sum_{v, h} \exp\big(-E(v, h)\big).
\]

La probabilidad marginal sobre las visibles es:

\[
P(v) = \sum_h P(v, h).
\]

En la práctica, calcular \(Z\) y las expectativas exactas suele ser
inviable para modelos medianos o grandes, por lo que se recurre a métodos de
aproximación.

---

## Aprendizaje

El objetivo del aprendizaje es ajustar \(W, b, c\) para maximizar la
probabilidad de los datos observados, es decir, maximizar la verosimilitud o
equivalentemente minimizar la energía media de los estados asociados a los
datos.

Las actualizaciones de parámetros involucran dos tipos de expectativas:

- **Fase positiva**: expectativas bajo la distribución condicionada por los
  datos reales.
- **Fase negativa**: expectativas bajo la distribución del modelo (muestreo
  libre).

De forma esquemática, para los pesos:

\[
\Delta W \propto \langle v h^T \rangle_{\text{datos}} -
\langle v h^T \rangle_{\text{modelo}}.
\]

Como las expectativas de la fase negativa requieren muestrear de la
distribución del modelo, se suelen utilizar **métodos de Monte Carlo** como
Gibbs sampling, a menudo con cadenas largas y medidas de equilibrio.

---

## Muestreo Gibbs y mezclado

En una BM general, el paso de Gibbs no es tan simple como en la RBM, porque las
unidades visibles (o las ocultas) no son condicionalmente independientes entre
sí. Esto implica que el muestreo puede requerir actualizar grupos de unidades
o realizar barridos más complejos.

Esto se traduce en:

- Costes computacionales más altos.
- Mayor dificultad para garantizar un buen mezclado de la cadena de Markov.

Por eso, en muchos sistemas prácticos se prefiere usar variantes restringidas
(RBM) o arquitecturas con estructura más controlada.

---

## Ventajas y limitaciones

### Ventajas

- Modelo muy flexible: permite conexiones densas entre nodos.
- Capaz de representar distribuciones complejas y multimodales.
- Base conceptual para modelos más avanzados (RBM, DBM, etc.).

### Limitaciones

- Entrenamiento costoso:
  - Cálculo aproximado de gradientes.
  - Muestreo lento.
- Difícil de escalar para datasets grandes sin técnicas adicionales.
- Menos estable y más difícil de depurar que arquitecturas restringidas.

---

## Uso conceptual en NeuroCampus

En NeuroCampus, la BM se entiende como la versión más general del modelo de
energía sobre las variables de evaluación docente. Sirve principalmente para:

- Explorar arquitecturas más flexibles.
- Comparar su comportamiento con modelos restringidos (RBM) y profundos (DBM).
- Entender el impacto de permitir conexiones adicionales entre variables.

La implementación concreta se encuentra en el módulo
`neurocampus.models.bm_manual`, donde se adapta esta teoría a código Python y
se integra con las estrategias de entrenamiento y evaluación.
