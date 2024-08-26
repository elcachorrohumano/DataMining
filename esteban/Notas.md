# Apuntes Minería y Análisis de datos

## Intro

### Estimación de decisión sobre de aprendizaje no supervisado

+ Al hacer clustering, se busca la distancia con cierto criterio de optimización.
  + El codo puede ayudar midiendo el nivel que optimiza la hetereogeneidad o el.
  + También la métrica algo silueta midiendo las distancias entre todo el cluster al centro y luego de cada elemento a otros clusters... esperando que sea menor a otros clusters.

### Aprendizaje por Refuerzo:

+ Una agente interactúa con un ambiente (robot o persona) y este toma decisions que optimicen ciertas cosas a pase de recompensas.
+ Se puede generar una matriz sobre las mejores acciones por cierto estado (Esta matriz, Q-table, producto de las x con las y´s).
  + Las y´s que se van encontrando pueden ser vistas como la función de costo. Al ir mejorando o empeorando conforme van cambiando las x´s.

### Problemas no lineales de ML

+ Se proyectan los datos a mayor dimensionalidad que termina siendo equivalente a reducir la dimensionalidad.
+ En caso de poder de cómputo limitado puede hacerse algo, en svm es el producto punto de todas las columnas, para reducir dimensionalidad antes de subrila.

### Hyperparámetros vs parámetros

Los parámetros (ajuste manual) se aprenden con los datos, los hiperparámetros se ajustan.

> $Aprender = Generalizar \ne Memorizar$

Cross validation :)

Regresión y clasificación se puede ver como el mismo problema: buscar el hiper plano (que separe o aproxime).

Siempre primero se busca salir del underfitting y luego se ajusta en caso de haber overfitting.

### Data Leakage (Esteban showing off he's bilingual)

Happens when our training data contains information about the target. This data will not be available at prediction time.

1. Target leakage
   - Predictors are registered after the target variable, e.g., if we include a time-saple as both predictor and target.
   - Predictors contarin (or is the same as) the target variable, e.g., if we include a time-sample as both predictor and target.
2. Train-test contamination
   - Happens when the same point, or a copy of it, exists in both the training and test sets. It might happen in large data bases.

### Datos numéricos

- Cuantitativos
- Positivos o negativos
- Discretos o contínuos

### Datos nominales (categóricos)

- Sin orden
- Nombres, labels, enumeraciones
- Pueden ser números, pero sin significado en sí
- No hay sentido en hacer operaciones sobre de ellos
- One-hot encoding
  - Estos datos no deben usarse (normalmente) directamente en una disminusión de dimensionalidad como el producto punto, pero puede transformarse para convertirlos en un número real
  - Si hay ordinales, sí hay un cierto orden pero habría que considerar que los saltos (como entre chico-mediano-grande) no sisempre son exactos

### Datos ordinales

- Tienen un orden, pero no es exactamente cuantificable. Como chico, mediano y grande.
- Los datos no nos dicen nada sin contexto.

## Teoría de la información

Rama de las matemáticas que nos permite cuantificar la cantidad de información en una señal.

- **Intuición**: Aprendiendo que un evento poco probable que sucede es más informativo que un evento probable que sucede.

  - Por ejemplo: no nos da mucha información la frase "El sol salió en la mañana"; por otro lado, "Hoy habrá un eclipse" sí nos aporta información.
- **Supuestos**:

  - Eventos muy probables dan poca información.
  - Cosas 100% probables no contienen información.
  - Eventos poco probables aportan mucha información.
  - Eventos independientes agregan información.
  - La longitud del mensaje debe ser proporcional a la cantidad de información.
- **Valores muy probables**:

  - Distribución one-hot en los posiblees valores.
  - Baja entropía (Mucho orden)
  - 1 evento --> nada de información
- **Valores poco probables**:

  - Distribución uniforme de los posibles valores.
  - Alta entropía (mucho desorden)
  - 1 evento --> mucha información
- **Posibilidad --> Información**

  - Un evento se ve como variable aleatoria.
  - La información (sorpresa) I(x) para la variable aleatoria x, con probabilidad p(x), se define como  la inversa de la probabilidad:
    - $I(x) = \frac{1}{p(x)}$
  - Si un evento es altamente probable, no hay sorpresa.
  - Si $p(x)=1 \rightarrow I(x)=1$... pero debería ser cero. Para resolver esto, utilizamos el logaritmo natural ya que $ln(1)=0$
    - $I(x) = ln(\frac{1}{p(x)})$
  - Si $p(x)=0 \rightarrow ln(\frac{1}{0})=ln(1)-ln(0)= indeterminado$
  - Normalmente, $ln(.)$ se usa (nats), ó $\log_2(x)$ (Bits)
- **Información mutua**:

  - Lo que podemos decir de $x$ al obervar $y$ o viceversa.
- Entropía: el valor esperado de la información.

  - $H(x) = E[I(x)] = - \sum P(x)log(P(x))$

## KDD

Knowledge Discovery in Data (KDD): is the overall process of collecting data, integrating and processing itm and developing methods and techniques for making sense of such data. What we often call data mining, might also be referred to as KDD, in chich case, Data Mining itself becomes but on step (the core) of the KDD process.

Ciclo:

1. Business understanding: definir el problema, requerimientos, métricas a optimizar,
2. Data understanding
3. Data preparation
4. Data validation
5. Modeling
6. Evaluation
7. Deployment

<img src="images/foto_para_callar_a_mau.png" alt="Example Image" width="400">

### Data analysis

EDA:

- Descriptive statistics: min, max, mean, std_dev.
- Distributions
- Entropy
- Correlations: queremos variables independientes sin correlación entre ellas pero con correlación con $y$.
- Variable selection (feature engineering).

### Deployment

- Visualize and present performance of out best model.
- Develop interpretation.
- Implement in production.
- Design maintainance cycle.

### CRISP-DM

```
<img src="images/crisp_steps.png" alt="Example Image" width="400">
```
