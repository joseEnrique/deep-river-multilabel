# Conclusiones del Estudio Experimental: La Estrategia de Loss Dependiente de la Capacidad

Este documento resume los hallazgos clave de los experimentos realizados sobre modelos LSTM para la detección de anomalías en el dataset AI4I. El estudio revela un fenómeno crucial que hemos denominado **"Estrategia de Loss Dependiente de la Capacidad"**: la función de pérdida óptima no es universal, sino que depende directamente de la capacidad de cómputo (tamaño del estado oculto) del modelo.

## 1. Contexto del Problema
El dataset presenta un **desbalanceo severo** (Clase 0 "Normal" >>> Clase 1 "Fallo"). Esto provoca que las funciones de pérdida estándar tiendan a converger hacia un mínimo local "cobarde", prediciendo siempre la clase mayoritaria para minimizar el error global, sacrificando la detección de fallos (F1 Score).

---

## 2. Escenario de Baja Capacidad (Hidden Dim 32): "Inteligencia sobre Fuerza Bruta" 🧠

En modelos con recursos limitados (**H32**), donde la red neuronal tiene poca memoria y capacidad de abstracción:

*   **El Ganador Indiscutible**: **RobustFocal Loss**.
*   **Rendimiento**: Alcanza un **Micro F1 del ~40%**, superando masivamente al Baseline BCE (~32%).
*   **Explicación**: El modelo pequeño es propenso a oscilaciones y a "rendirse" ante la dificultad de la clase minoritaria. `RobustFocal` actúa como un **estabilizador inteligente**:
    1.  Amortigua los gradientes explosivos al inicio del entrenamiento.
    2.  Focaliza la atención del modelo en los ejemplos difíciles pero "aprendibles", ignorando los outliers imposibles.
    3.  Evita que el modelo caryga en la predicción trivial de la clase 0.

**Conclusión**: Cuando el hardware es limitado (Edge Computing, IoT), la inteligencia algorítmica de **RobustFocal** es esencial para compensar la falta de capacidad del modelo.

---

## 3. Escenario de Alta Capacidad (Hidden Dim 128): "La Estabilidad Gana a la Complejidad" 💪

En modelos con alta capacidad (**H128**), donde la red puede aprender patrones complejos sin saturarse:

*   **Los Ganadores**: Empate técnico entre **FullAdaptive Loss** y **RobustFocal Loss**.
*   **Rendimiento**: Alcanzan el pico máximo del estudio con un **Macro F1 del ~43.9%**.
*   **Explicación**: Cuando el modelo tiene suficientes recursos (memoria y neuronas), las heurísticas estables que ajustan la pérdida basándose en el rendimiento del momento (Recall y Accuracy) son extremadamente efectivas y consistentes. La capacidad del modelo permite aprovechar al máximo estas señales dinámicas sin desestabilizarse.

**El Fracaso de la Optimización Directa (Learnable Loss):**
A pesar de la alta capacidad del modelo, permitir que la red aprenda sus propios parámetros de loss ($\alpha$ y $\gamma$) mediante gradiente descendente resulta en un rendimiento significativamente inferior (**~38.1% Macro F1**).
*   **Causa**: El efecto de **"Target Moving"** (objetivo móvil). El modelo intenta aprender a detectar anomalías y, simultáneamente, intenta aprender cómo debe evaluarse a sí mismo. Esta doble tarea genera ruido constante en la señal de gradiente, impidiendo la convergencia a los niveles óptimos que logran las heurísticas estables.

---

## 4. Análisis de Eficiencia Computacional ⏱️

*   **BCE Baseline**: Es la opción más rápida por ser estática (**~25-29ms/step**).
*   **RobustFocal / FullAdaptive**: Añaden un coste computacional secundario (**~35-40ms/step**). Requieren el cálculo de estadísticas del batch, actualización de estados internos y logaritmos complejos. Sin embargo, en un entorno de streaming, este coste menor en tiempo está más que justificado por el aumento drástico en las métricas de detección.

---

## 5. El Fracaso de la Adaptación Bidireccional 📉

La arquitectura `BidirectionalAdaptiveLoss`, diseñada para ajustar dinámicamente Alpha y Gamma hacia un objetivo fijo (Target Recall/Score), no logró superar a las estrategias puramente dinámicas (RobustFocal o FullAdaptive).
*   **Causa**: **Sobre-ingeniería**. Al igual que con *Learnable Loss*, forzar parámetros hacia objetivos rígidos en un entorno de streaming (donde la distribución puede cambiar) introduce inestabilidades y dificulta que el modelo encuentre su flujo natural de aprendizaje.

---

## Resumen Final

| Escenario de Uso | Estrategia Recomendada | Razón Principal |
| :--- | :--- | :--- |
| **Dispositivos Edge / IoT (Recursos Limitados - H32)** | **RobustFocal Loss** | Máxima estabilización en modelos pequeños propensos a divergir. |
| **Cloud / Servidores (Alto Rendimiento - H128)** | **FullAdaptive / RobustFocal** | La alta capacidad del modelo explota al máximo el ajuste fino en tiempo real sin requerir parámetros aprendibles inestables. |
| **Estabilidad / Caso General** | **RobustFocal Loss** | Es la opción más segura ("Safe Bet"). Nunca es la peor opción y siempre es competitiva. |

---

## 6. Anexo: Formulación Matemática de las Estrategias Clave 📐

Para sustentar estas conclusiones, detallamos la formulación matemática de las tres funciones más relevantes del estudio.

### A. Binary Cross Entropy (BCE) - El Baseline
Es la función estándar para clasificación binaria. Penaliza logarítmicamente el error entre la predicción y la etiqueta real.

$$ L_{BCE}(p, y) = - (y \cdot \log(p) + (1 - y) \cdot \log(1 - p)) $$

*   $y \in \{0, 1\}$: Etiqueta real.
*   $p \in [0, 1]$: Probabilidad predicha por el modelo.

**Comportamiento**: Trata todas las clases por igual. En datasets desbalanceados, el término $(1-y)\log(1-p)$ (clase mayoritaria) domina la suma total del gradiente, empujando al modelo a predecir siempre 0.

---

### B. Weighted Cross Entropy (WCE) - La Fuerza Bruta
Introduce un coeficiente de ponderación $\beta$ para la clase positiva (fallos).

$$ L_{WCE}(p, y) = - (\beta \cdot y \cdot \log(p) + (1 - y) \cdot \log(1 - p)) $$

*   $\beta > 1$: Peso asignado a la clase positiva. En nuestros experimentos, $\beta$ se ajusta inversamente proporcional a la frecuencia de la clase.

**Comportamiento**: Amplifica artificialmente el error cuando el modelo falla en detectar un caso positivo.
*   **Ventaja**: Fuerza al modelo a prestar atención a la minoría.
*   **Riesgo**: Si $\beta$ es muy alto, puede causar inestabilidad o incrementar demasiados los Falsos Positivos. Requiere un modelo con capacidad suficiente para manejar esta señal distorsionada sin romperse (por eso funciona bien en H128).

---

### C. Robust Focal Loss - La Inteligencia Adaptativa
Una evolución de la Focal Loss diseñada para ser resistente a outliers y estabilizar el entrenamiento.

$$ L_{RFL}(p_t) = - \alpha_t (1 - p_t)^{\gamma_t} \log(p_t) $$

Donde $p_t$ es la probabilidad de la clase verdadera:
$$ p_t = \begin{cases} p & \text{si } y=1 \\ 1-p & \text{si } y=0 \end{cases} $$

**Mecanismo Adaptativo**:
A diferencia de la Focal Loss estática, $\alpha_t$ y $\gamma_t$ no son constantes, sino que evolucionan suavemente usando medias móviles (Momentum) basadas en el rendimiento del modelo:

1.  **Alpha Dinámico (Balanceo)**: Se ajusta inversamente al **Recall** (Sensibilidad).
    $$ \alpha_{t+1} = m \cdot \alpha_t + (1-m) \cdot (1 - \text{Recall}_{batch}) $$
    *   *Si el Recall baja, Alpha sube*: El modelo presta más atención a la clase positiva.

2.  **Gamma Dinámico (Dificultad)**: Se ajusta inversamente a la **Exactitud (Accuracy)**.
    $$ \gamma_{t+1} = m \cdot \gamma_t + (1-m) \cdot (1 - \text{Accuracy}_{batch}) $$
    *   *Si la Exactitud baja, Gamma sube*: El modelo se focaliza más en los ejemplos difíciles.

**Comportamiento**:
*   El término $(1 - p_t)^\gamma$ reduce el peso de los ejemplos fáciles (bien clasificados), centrando el gradiente en los difíciles.
*   El suavizado por momentum evita oscilaciones bruscas, lo que es vital para estabilizar modelos pequeños (H32) que de otra forma divergirían.
