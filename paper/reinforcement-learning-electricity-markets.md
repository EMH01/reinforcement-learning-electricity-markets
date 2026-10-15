# Aprendizaje por refuerzo aplicado a mercados eléctricos: diseño de entornos, algoritmos, herramientas y criterios de evaluación

**Autora:** Esther M. Martin Hernández  
**Tipo:** Informe técnico de investigación  
**Versión:** 1.0 · octubre de 2026

# Resumen

Los mercados eléctricos combinan incertidumbre, restricciones operativas, información que evoluciona en el tiempo y decisiones con consecuencias económicas acumulativas. Estas características han impulsado el uso del aprendizaje por refuerzo (*reinforcement learning*, RL) como marco para estudiar problemas de decisión secuencial. Su aplicación, sin embargo, depende tanto del diseño del entorno como del algoritmo elegido: es necesario definir qué información observa el agente, qué acciones puede ejecutar, cómo se representa el objetivo, qué restricciones deben respetarse y cómo se evalúa el comportamiento fuera de muestra. Este informe reúne y organiza los principales elementos metodológicos para diseñar y analizar sistemas de RL aplicados a mercados eléctricos. Se revisan la formulación mediante procesos de decisión de Markov, los espacios de observación y acción, las funciones de recompensa, el tratamiento de restricciones, las familias de algoritmos más utilizadas y las herramientas de implementación y simulación. También se sintetizan patrones de aplicación presentes en la literatura, incluyendo estrategias de oferta, activos flexibles, generación renovable, decisiones continuas y entornos multiagente. La revisión pone especial énfasis en la prevención de fuga de información, la fidelidad del simulador, el uso de baselines, la evaluación temporal y la trazabilidad experimental.

**Palabras clave:** aprendizaje por refuerzo; mercados eléctricos; trading energético; diseño de entornos; simulación; backtesting temporal; gestión de riesgo; decisión secuencial.

# 1. Introducción

El aprendizaje por refuerzo aborda problemas en los que un agente observa información del entorno, selecciona acciones y recibe señales de recompensa con el objetivo de maximizar el retorno acumulado esperado. La formulación clásica se apoya en procesos de decisión de Markov (MDP), que proporcionan un marco para representar estados, acciones, transiciones y recompensas [1](#ref-1). En mercados eléctricos, esta estructura resulta pertinente porque las decisiones se toman bajo incertidumbre, dependen de información disponible en cada instante y pueden afectar a resultados posteriores.

La literatura ha mostrado un crecimiento sostenido de aplicaciones de RL a estrategias de oferta, despacho, almacenamiento, arbitraje, gestión de activos y control de riesgo en mercados liberalizados [2](#ref-2), [3](#ref-3). La transferencia desde un experimento académico a un sistema útil exige, no obstante, resolver varios problemas metodológicos: evitar fuga de información futura, representar con suficiente fidelidad las reglas del entorno, imponer restricciones técnicas y económicas y evaluar la política en periodos no utilizados durante el entrenamiento.

Este informe organiza esos elementos con un objetivo práctico: disponer de un marco general para diseñar y evaluar aplicaciones de RL en mercados eléctricos. Para ello, cubre cuatro frentes: (i) formalizar el entorno de decisión y sus componentes; (ii) ordenar las principales familias de algoritmos y arquitecturas; (iii) identificar herramientas de implementación y simulación; y (iv) sintetizar criterios de evaluación y patrones de aplicación descritos en la literatura.

# 2. Alcance y naturaleza de la revisión

El documento se plantea como una **revisión narrativa y técnica orientada al diseño**, no como una revisión sistemática. Se priorizan trabajos fundacionales de RL, artículos revisados por pares, publicaciones de software científico y documentación técnica relevante. Los ejemplos aplicados se utilizan para identificar patrones metodológicos recurrentes en problemas de oferta, control continuo, almacenamiento, generación renovable, gestión de activos y entornos multiagente.

Por tanto, las conclusiones deben leerse como una síntesis técnica, no como un metaanálisis exhaustivo de toda la literatura. Cada fuente se utiliza para respaldar la afirmación concreta a la que se refiere, sin extrapolar resultados de un entorno experimental a otro cuando no hay base para hacerlo.

## 2.1. Fuentes y alcance

El informe se ha elaborado a partir de literatura científica, documentación oficial y fuentes públicas.

# 3. Fundamentos y diseño del entorno de aprendizaje por refuerzo

## 3.1. Formulación como proceso de decisión secuencial

Un MDP puede expresarse como la tupla

**M = (S, A, P, R, γ),**

donde *S* es el espacio de estados, *A* el espacio de acciones, *P* la dinámica de transición, *R* la función de recompensa y *γ* el factor de descuento [1](#ref-1). En cada instante de decisión *t*, el agente recibe una observación *o_t* —que puede coincidir o no con el estado completo—, elige una acción *a_t* y recibe una recompensa *r_t*. El objetivo es aprender una política π que maximice el retorno esperado:

**G_t = Σ(k=0…∞) γ^k r_(t+k+1).**

En trading eléctrico, la hipótesis de observabilidad completa rara vez es estrictamente válida: el agente desconoce el futuro y puede no observar el libro de órdenes completo, la estrategia de otros participantes o determinadas restricciones no observables o no publicadas. Por ello, es más preciso hablar de **observación** cuando el vector entregado al agente representa solo la información disponible en ese instante.

## 3.2. Espacio de estado u observación

El estado operativo puede organizarse en cuatro bloques.

**a) Contexto de mercado y variables exógenas.** Incluye precios observados, señales de precio, profundidad o liquidez cuando esté disponible, previsiones de generación y demanda, meteorología, indisponibilidades y otras señales externas. Estas variables no son controladas por el agente, pero modifican el valor esperado de sus acciones.

**b) Estado interno de la posición o del activo.** En un problema de trading sin activo físico puede incluir posición acumulada, energía pendiente, exposición acumulada o límites de actuación. Cuando existe almacenamiento o generación gestionable, se añaden variables como estado de carga, potencia disponible, rampas, costes marginales y restricciones técnicas.

**c) Información de decisiones previas.** En problemas secuenciales puede ser necesario representar acciones ya ejecutadas, compromisos acumulados, capacidad disponible o exposición existente. Sin este bloque el agente puede tratar cada decisión como independiente aunque forme parte de una misma trayectoria.

**d) Codificación temporal.** Hora, día de la semana, estacionalidad y tiempo restante hasta un evento relevante pueden codificarse de forma numérica o cíclica. La elección debe respetar estrictamente el principio *point-in-time*: ninguna variable puede contener información que no estuviera disponible en el instante en que se simula la decisión.

La inclusión de memoria explícita mediante redes recurrentes o extractores temporales puede ser útil cuando la observación instantánea no resume toda la información relevante. No obstante, una arquitectura más compleja solo está justificada si mejora de forma reproducible el desempeño fuera de muestra; no debe asumirse que una LSTM o un Transformer será superior por defecto.

## 3.3. Espacio de acciones

El espacio de acciones debe reflejar con la menor ambigüedad posible qué decisiones son realmente ejecutables.

En un **espacio discreto**, la política puede seleccionar entre acciones como comprar, vender o no operar, o elegir entre un conjunto finito de estrategias. Este diseño facilita el entrenamiento y la interpretación, pero puede perder precisión si la magnitud de la posición es económicamente relevante.

En un **espacio continuo**, la acción puede representar volumen, variación de posición, potencia de carga/descarga o parámetros de precio. DDPG, TD3 y SAC fueron desarrollados específicamente para control continuo [8](#ref-8)–[10](#ref-10), mientras que PPO también se utiliza con frecuencia en espacios continuos mediante políticas estocásticas [7](#ref-7).

Una tercera alternativa es la **acción estructurada**, por ejemplo un vector de bloques precio–cantidad o una combinación de volumen y agresividad de ejecución. Este diseño es más próximo a la operativa de mercado, pero requiere un simulador capaz de traducir la acción en una probabilidad o resultado de casación. Cuando esa capa no existe, atribuir a la política una capacidad de ejecución que el entorno no modela produce una estimación optimista del rendimiento.

## 3.4. Función de recompensa

La recompensa conecta el objetivo del problema con el comportamiento que aprende la política. Una formulación genérica puede escribirse como:

**r_t = ΔJ_t − λ_risk C_risk,t − λ_op C_op,t − λ_cost C_cost,t,**

donde ΔJ representa el resultado incremental asociado al objetivo, *C_risk* un término de riesgo, *C_op* penalizaciones por acciones operativamente inviables y *C_cost* otros costes relevantes para el entorno modelado.

Los términos incluidos deben tener una interpretación económica u operativa clara. Si se incorporan medidas de riesgo de cola, el Value at Risk (VaR) resume un cuantil de pérdidas, mientras que el Conditional Value at Risk (CVaR) cuantifica la pérdida media en la cola más desfavorable. El CVaR dispone de una formulación de optimización bien establecida [4](#ref-4) y ha sido utilizado en trabajos de decisión secuencial sensible al riesgo [5](#ref-5).

En algunos problemas resulta útil definir la recompensa de forma incremental respecto de una política de referencia. Este enfoque facilita atribuir la mejora al comportamiento aprendido y evita que el agente sea premiado por componentes del resultado que no dependen de sus decisiones.

## 3.5. Restricciones: penalización frente a proyección

Las restricciones pueden tratarse de tres maneras. La primera es penalizar violaciones dentro de la recompensa. La segunda es **recortar o proyectar** la acción a la región factible antes de ejecutarla. La tercera es diseñar el espacio de acciones de modo que solo genere decisiones válidas. En aplicaciones industriales, las dos últimas opciones son preferibles para restricciones duras —límites de potencia, energía, posición o mercado—, mientras que las penalizaciones son apropiadas para preferencias blandas como reducir rotación, exposición o riesgo.

Debe distinguirse entre la acción propuesta por la política y la acción finalmente ejecutada por el entorno. Esta trazabilidad es necesaria para auditar por qué una política obtiene un resultado determinado y para evitar que el agente aprenda a explotar errores de modelado.

# 4. Algoritmos y arquitecturas relevantes

## 4.1. Métodos basados en valor, política y actor–critic

Los métodos **basados en valor** estiman funciones como *V(s)* o *Q(s,a)* y derivan la decisión a partir de esas estimaciones. DQN demostró la capacidad de combinar redes neuronales y Q-learning en espacios de acción discretos [6](#ref-6). Las arquitecturas *dueling* separan el valor del estado y la ventaja relativa de las acciones, lo que puede mejorar la estimación cuando varias acciones tienen efectos similares [12](#ref-12).

Los métodos **basados en política** optimizan directamente una distribución de acciones. PPO limita el tamaño efectivo de las actualizaciones mediante un objetivo sustituto con *clipping*, buscando un equilibrio entre estabilidad y simplicidad [7](#ref-7).

Los métodos **actor–critic** combinan una política (actor) y una estimación de valor (critic). DDPG extiende ideas de gradiente de política determinista a acciones continuas [8](#ref-8). TD3 introduce doble crítico, objetivos suavizados y actualizaciones retrasadas para reducir sesgos de sobreestimación [9](#ref-9). SAC adopta un objetivo de máxima entropía y aprendizaje *off-policy*, favoreciendo exploración y reutilización de experiencia [10](#ref-10). A2C es una variante síncrona de la familia actor–critic asociada al trabajo que popularizó A3C; no debe confundirse con “Asynchronous Advantage Actor-Critic”, denominación que corresponde a A3C [11](#ref-11).

## 4.2. Online RL y offline RL

En **online RL**, la política genera nueva experiencia al interactuar con un entorno real o simulado. PPO es *on-policy*: aprende principalmente con trayectorias producidas por la política actual o una versión muy próxima [7](#ref-7). SAC y TD3 son *off-policy* y pueden reutilizar transiciones almacenadas en un *replay buffer* [9](#ref-9), [10](#ref-10).

En **offline RL**, el aprendizaje se realiza a partir de un dataset estático sin interacción adicional durante el entrenamiento. La dificultad principal es el *distribution shift*: la política puede proponer acciones poco representadas en el dataset y el estimador de valor puede asignarles valores excesivamente optimistas. CQL aborda este problema mediante estimaciones conservadoras de Q [13](#ref-13), mientras que IQL evita evaluar explícitamente acciones fuera de la distribución mediante una formulación implícita [14](#ref-14). AWAC es especialmente relevante cuando se desea aprovechar datos previos y continuar posteriormente con ajuste online [15](#ref-15).

Es importante no confundir **offline RL** con entrenar un algoritmo online dentro de un simulador alimentado por datos históricos. En el segundo caso, el agente sí interactúa y genera transiciones en un entorno simulado; que las series exógenas procedan del histórico no convierte automáticamente el método en offline RL.

## 4.3. Arquitecturas neuronales y memoria temporal

Cuando el estado es un vector tabular de variables, una MLP constituye un baseline razonable por su simplicidad. Las redes recurrentes pueden aportar memoria cuando la observación es parcial o existe dependencia temporal no resumida por el estado. Los Transformers permiten modelar relaciones de largo alcance y se han incorporado a sistemas de trading y almacenamiento, pero su uso debe justificarse mediante comparación experimental frente a arquitecturas más sencillas.

Las CNN son adecuadas cuando la entrada posee estructura espacial o matricial; por ejemplo, mapas o representaciones densas del libro de órdenes. Los autoencoders pueden emplearse como reducción de dimensionalidad, pero añaden otra etapa de aprendizaje cuyo beneficio debe demostrarse.

## 4.4. Aprendizaje multiagente y jerárquico

En un entorno con participantes estratégicos, el problema puede formularse como aprendizaje multiagente. MADDPG introdujo un esquema de entrenamiento centralizado y ejecución descentralizada para entornos cooperativos y competitivos [16](#ref-16). MAPPO mostró que variantes de PPO pueden constituir baselines muy competitivos en tareas cooperativas [17](#ref-17).

El **modelado basado en agentes (ABM)** y el **aprendizaje multiagente (MARL)** están relacionados, pero no son sinónimos. ABM es un paradigma de simulación de sistemas formados por entidades autónomas con reglas de comportamiento; MARL se refiere específicamente a agentes que aprenden políticas mediante RL. Un ABM puede contener agentes no aprendices, y un sistema MARL no necesita adoptar toda la metodología típica de ABM.

El aprendizaje jerárquico introduce decisiones en diferentes escalas temporales —por ejemplo, una política de alto nivel que selecciona un modo operativo y otra de bajo nivel que decide volúmenes—. Aunque resulta atractivo para problemas complejos, incrementa la dificultad de entrenamiento y de atribución causal del rendimiento, por lo que conviene utilizarlo cuando exista una jerarquía operacional real.

## 4.5. Aprendizaje por imitación y regularización con el histórico

El histórico operativo puede utilizarse para algo más que comparar resultados. El *behavioral cloning* aprende una política supervisada a partir de pares estado–acción, proporcionando una inicialización plausible, aunque por sí solo reproduce el comportamiento observado y no optimiza explícitamente una recompensa.

DAgger reduce el error de distribución de la imitación supervisada agregando de manera iterativa estados visitados por la política y consultando la acción de un experto [18](#ref-18). GAIL aprende una política cuya distribución de trayectorias se aproxima a la de las demostraciones mediante un esquema adversarial [19](#ref-19), mientras que AIRL busca recuperar una función de recompensa más interpretable y transferible [20](#ref-20). En trading energético, estas técnicas son útiles principalmente como inicialización, restricción de soporte o benchmark; no sustituyen la evaluación económica fuera de muestra.

# 5. Implementación técnica y herramientas

## 5.1. Interfaces de entorno

La separación entre agente y entorno es crítica para la reproducibilidad. Gymnasium mantiene una API ampliamente utilizada para entornos de RL y formaliza métodos como `reset()` y `step()` [21](#ref-21). Para problemas multiagente, PettingZoo define interfaces específicas para interacción simultánea o secuencial entre múltiples agentes [22](#ref-22).

Esta separación permite cambiar el algoritmo sin reescribir la lógica de mercado, siempre que se mantengan estables las observaciones, acciones, transiciones y recompensas.

## 5.2. Frameworks de aprendizaje

**Tabla 1. Frameworks relevantes para prototipado e investigación**

| Framework | Uso principal | Observación |
|---|---|---|
| Stable-Baselines3 | Baselines model-free (PPO, SAC, TD3, DQN, A2C) | Adecuado para prototipado reproducible con PyTorch [23](#ref-23). |
| RLlib | Entrenamiento distribuido y multiagente | Proporciona abstracciones para escalar experimentos y políticas [24](#ref-24). |
| CleanRL | Implementaciones compactas | Útil para auditoría del algoritmo y experimentos controlados [25](#ref-25). |
| MushroomRL | Investigación modular | Facilita experimentar con familias model-free y model-based [26](#ref-26). |
| TorchRL | Componentes RL dentro de PyTorch | Apropiado para pipelines personalizados, datasets y objetivos de RL [27](#ref-27). |
| Tianshou | Colección modular de algoritmos | Ofrece componentes reutilizables y soporte para múltiples paradigmas [28](#ref-28). |

En la práctica, la elección del framework depende de la reproducibilidad que se necesite, los algoritmos disponibles, la trazabilidad de los experimentos y la facilidad de integración con el resto del sistema.

## 5.3. Simulación de mercado, activos y red

Un entorno de RL para mercados eléctricos puede separarse conceptualmente en varias capas: **mercado**, **activo o estado interno**, **ejecución y resultado económico** y **datos exógenos**. La capa de mercado define las reglas relevantes para la decisión; la capa de activo aplica restricciones físicas cuando existen; la ejecución traduce la acción del agente en una consecuencia observable; y los datos exógenos aportan la información disponible en cada instante.

ASSUME ofrece un marco abierto de simulación basada en agentes para mercados eléctricos e integra estrategias con aprendizaje por refuerzo [29](#ref-29). Para modelado físico, PyPSA permite análisis y optimización de sistemas energéticos [30](#ref-30), pandapower facilita flujo de potencia y otras funciones de análisis de redes eléctricas [31](#ref-31), y MATPOWER continúa siendo una referencia abierta para estudios de flujo de potencia y *optimal power flow* [32](#ref-32). Estas herramientas no son intercambiables: su utilidad depende de si el problema requiere representar reglas de mercado, restricciones físicas, interacción entre agentes o una combinación de estas capas.

Un simulador útil no necesita reproducir todo el sistema eléctrico. Debe incluir únicamente los mecanismos que afectan a las decisiones analizadas y documentar de forma explícita las simplificaciones adoptadas. Añadir componentes que no modifican el comportamiento del agente incrementa la complejidad sin mejorar necesariamente la validez del experimento.

# 6. Datos, reglas de mercado y validez del entorno

La calidad de un experimento de RL depende de que las observaciones reflejen la información disponible en el momento de la decisión. Los mercados eléctricos ofrecen distintas fuentes de datos públicos —por ejemplo, precios, resultados agregados, variables de demanda, generación o meteorología—, pero la cobertura y el nivel de detalle varían entre mercados y periodos. La existencia de información agregada no implica disponer de órdenes individuales, estrategias de participantes ni condiciones de ejecución completas.

La resolución temporal del entorno debe responder al fenómeno que se desea estudiar y a la información realmente disponible. Elegir una granularidad más fina no mejora por sí sola el modelo si los datos o las reglas no permiten reconstruir de forma fiable las transiciones entre decisiones. Del mismo modo, cualquier transformación, previsión o variable derivada debe respetar el principio *point-in-time*.

Las reglas de mercado también forman parte del entorno. Cuando cambian, pueden modificar el significado de una observación, una acción o una función de coste. Por ello, la versión de las reglas, los datos y la configuración del simulador deben registrarse junto con cada experimento [2](#ref-2), [3](#ref-3).

En el ámbito ibérico existe información pública que puede alimentar entornos de investigación. OMIE publica precios y resultados del mercado diario, curvas agregadas de oferta y demanda y resultados de mercados intradiarios, entre otros ficheros y visualizaciones [[33]](#ref-33). No obstante, disponer de curvas agregadas no equivale a disponer del libro de órdenes individual ni de la intención de cada participante; cualquier modelo de ejecución debe respetar esa limitación.

La granularidad temporal es también parte de la validez del entorno. La regulación española estableció un periodo de liquidación de desvíos de 15 minutos desde el 1 de diciembre de 2024 y adaptó posteriormente los procedimientos a la negociación cuarto-horaria en los mercados diario e intradiario [[34]](#ref-34). Por tanto, una simulación que represente la operación española actual debe alinear productos, posiciones y liquidaciones con esa granularidad cuando corresponda.

# 7. Casos de uso en mercados eléctricos y trading operativo

Los siguientes trabajos se seleccionan porque representan problemas transferibles a sistemas de decisión en mercados eléctricos: *bidding*, acciones continuas, operación multi-mercado, BESS, intradiario y simulación multiagente.

## 7.1. Ofertas estratégicas y generación de curvas de oferta en day-ahead

- **Referencia principal**  
  Di Persio, Garbelli y Giordano, *Energy Economics* (2025) [[35]](#ref-35).

- **Objetivo**  
  Construir curvas de oferta escalonadas precio–cantidad para un participante en el mercado diario, maximizando el beneficio bajo restricciones de producción.

- **Enfoque**  
  El trabajo utiliza DDPG, un algoritmo actor–critic para acción continua, precisamente porque la curva de oferta se representa como un vector de alta dimensión. Esto conecta directamente con lo descrito en **3.3 — Espacio de acciones** y **4.1 — Métodos basados en valor, política y actor–critic**.

- **Estado / información de entrada**  
  El modelo utiliza información histórica de precios del mercado eléctrico italiano y parámetros relacionados con costes y capacidad para producir la curva de oferta.

- **Acción**  
  Una curva de oferta escalonada, no una única decisión binaria.

- **Restricciones**  
  La política debe respetar límites de producción y la estructura de la oferta.

- **Reward**  
  Se basa en el beneficio generado por la estrategia. Los autores normalizan el resultado respecto a un máximo teórico definido bajo las hipótesis del experimento.

- **Evaluación y resultados**  
  El trabajo reporta hasta aproximadamente un **85 % de recompensa normalizada** en sus experimentos [[35]](#ref-35).

- **Aprendizajes transferibles**
  - Las acciones precio–cantidad pueden representarse como control continuo.
  - El *bidding* estratégico necesita una capa explícita de casación.
  - Es útil separar una primera fase *price-taker* de una segunda fase donde el precio de la acción sí afecte a la ejecución.

## 7.2. Estrategias multi-mercado con BESS y aprendizaje temporal

- **Referencia principal**  
  Li, Wang, Zhang y Wang, *IEEE Transactions on Energy Markets, Policy and Regulation* (2024) [[36]](#ref-36).

- **Objetivo**  
  Optimizar conjuntamente la participación de un BESS en el mercado spot y en mercados australianos de contingencia FCAS.

- **Arquitectura**  
  El método TempDRL combina:
  - un *Transformer-based Temporal Feature Extractor* (TTFE);
  - un agente Soft Actor-Critic (SAC) para el *bidding* continuo [[36]](#ref-36).

  Esto conecta con **4.3 — Arquitecturas neuronales y memoria temporal** y **4.1 — Métodos basados en valor, política y actor–critic**.

- **Estado**
  - precios de múltiples mercados;
  - representación temporal extraída por el TTFE;
  - estado de carga del BESS.

- **Acción**  
  Decisiones continuas de *bidding* y asignación entre mercado spot y servicios FCAS.

- **Restricciones**  
  El BESS debe respetar capacidad, potencia y compatibilidad entre energía reservada para distintos servicios.

- **Reward**  
  Combina los ingresos obtenidos por las distintas vías de mercado y las consecuencias operativas de la decisión.

- **Datos y evaluación**  
  Se utilizan precios realistas del Australian National Electricity Market. El artículo compara el enfoque con estrategias basadas en optimización y con otros métodos DRL, reportando una mejora relevante y mayor rentabilidad de la participación conjunta frente a operar mercados de forma aislada [[36]](#ref-36).

- **Aprendizajes transferibles**
  - El estado interno del activo condiciona las decisiones futuras.
  - Las decisiones multi-mercado generan coste de oportunidad entre mercados.
  - La información temporal puede aportar valor, pero debe compararse contra MLP simples antes de justificar arquitecturas más complejas.

## 7.3. Estrategias multi-mercado para renovables con almacenamiento

- **Referencia principal**  
  Cardo-Miota *et al.*, *Applied Energy* (2025) [[37]](#ref-37).

- **Objetivo**  
  Maximizar ingresos de un sistema renovable + BESS participando simultáneamente en el mercado day-ahead y en servicios de reserva del programa DS3 en Irlanda.

- **Enfoque**  
  El problema se formula como MDP y utiliza TD3 como algoritmo actor–critic de acción continua [[37]](#ref-37).

- **Estado**
  - señales y precios de mercado;
  - producción renovable;
  - estado del almacenamiento.

- **Datos**  
  El caso de estudio utiliza datos históricos de una planta fotovoltaica de 7 MW y un BESS de 1 MWh [[37]](#ref-37).

- **Acción**  
  Decisiones continuas de uso del almacenamiento y participación en mercados.

- **Restricciones**
  - capacidad del BESS;
  - potencia;
  - SoC;
  - compatibilidad con los servicios de mercado.

- **Reward**  
  Resultado económico agregado de la participación multi-mercado, sujeto a las restricciones del activo.

- **Evaluación**  
  El artículo compara el agente con estrategias de referencia y reporta beneficios económicos adicionales en los escenarios estudiados [[37]](#ref-37).

- **Aprendizajes transferibles**
  - Las restricciones físicas deben formar parte del entorno, no añadirse después.
  - Conviene reportar resultado por mercado para entender de dónde procede la mejora.
  - El valor de una política multi-mercado debe medirse frente a un baseline con información comparable.

## 7.4. Intradiario continuo para renovables: beneficio y penalización por desvío

- **Referencia principal**  
  Abate, Liu, Liu y Zhang (2025) [[38]](#ref-38).

- **Objetivo**  
  Optimizar el trading intradiario continuo de una planta fotovoltaica bajo incertidumbre de generación y precios, equilibrando beneficio de trading y coste de desbalance.

- **Algoritmo**  
  El trabajo utiliza PPO con una política deliberadamente interpretable [[38]](#ref-38).

- **Estado**  
  Combina variables de mercado, microestructura, información histórica y señales relacionadas con la producción y el *forecast*.

  Esto es coherente con los bloques propuestos en **3.2 — Espacio de estado u observación**.

- **Acción**  
  El agente ajusta la estrategia de trading de forma secuencial a medida que cambia la información disponible.

- **Reward**
  - beneficio de trading;
  - penalización asociada al desbalance.

  Esta formulación es especialmente relevante para **3.4 — Función de recompensa**, porque evita optimizar beneficio bruto ignorando el coste final de una posición residual.

- **Evaluación**  
  El modelo se entrena con históricos y se evalúa *out-of-sample*. El trabajo reporta mejora consistente frente a los benchmarks utilizados [[38]](#ref-38).

- **Aprendizajes transferibles**
  - El intradiario puede entenderse como mecanismo de ajuste progresivo de posición.
  - Reward y desvío deben modelarse conjuntamente.
  - La evaluación temporal fuera de muestra es obligatoria.
  - Una política más interpretable puede facilitar validación operativa y gobierno del modelo.

## 7.5. Multiagente, simulación estratégica y mercados locales

- **Referencia principal**  
  Bâra y Oprea, *Systems Research and Behavioral Science* (2026) [[39]](#ref-39).

- **Objetivo**  
  Modelar trading en mercados eléctricos locales con prosumidores heterogéneos, integrando generación, consumo flexible, vehículos eléctricos y calefacción.

- **Arquitectura**  
  El trabajo combina:
  - Mesa para ABM;
  - PyMarket para el mecanismo de mercado;
  - PPO de Stable-Baselines3 como política de *bidding* [[39]](#ref-39).

  Esta combinación ilustra la diferencia explicada en **4.4 — Aprendizaje multiagente y jerárquico**: el ABM representa el ecosistema de agentes y RL optimiza decisiones dentro de ese ecosistema.

- **Estrategias comparadas**
  - **S1:** el agente determina principalmente el precio de puja.
  - **S2:** el agente optimiza precio y cantidad y coordina flexibilidad de cargas [[39]](#ref-39).

- **Datos y simulación**  
  El artículo utiliza una comunidad de 114 hogares, datos de consumo a resolución de 15 minutos y simulación de un año, con DAM e IDM locales [[39]](#ref-39).

- **Reward**  
  Se calcula a partir del resultado de mercado después del *clearing* y guía a PPO hacia mejores decisiones de *bidding*.

- **Métricas**  
  El trabajo utiliza KPIs de comunidad y ahorro financiero, y reporta que S2 supera a S1. En verano, por ejemplo, el ahorro financiero reportado pasa de 11,58 % a 17,88 %; en invierno, de 4,21 % a 5,38 % [[39]](#ref-39).

- **Aprendizajes transferibles**
  - Es una referencia útil para entender interacción entre agentes, *clearing* y flexibilidad.
  - No debe trasladarse directamente a OMIE: la escala, reglas y tipo de participantes son diferentes.
  - Su principal valor metodológico aparece cuando se desea abandonar el supuesto *price-taker*.

## 7.6. Síntesis comparativa de los casos

**Tabla 2. Casos publicados y elementos metodológicos**

| Caso | Activo / mercado | Algoritmo | Acción | Principal aprendizaje transferible |
|---|---|---|---|---|
| Di Persio *et al.* [[35]](#ref-35) | Day-ahead | DDPG | Curva precio–cantidad | El *bidding* estructurado necesita simular casación |
| Li *et al.* [[36]](#ref-36) | BESS spot + FCAS | SAC + Transformer | Continua multi-mercado | Estado temporal + SoC + coste de oportunidad |
| Cardo-Miota *et al.* [[37]](#ref-37) | PV+BESS, day-ahead + reservas | TD3 | Operación continua | Restricciones físicas dentro del entorno |
| Abate *et al.* [[38]](#ref-38) | PV, intradiario continuo | PPO | Ajuste secuencial | Beneficio + desbalance + evaluación *out-of-sample* |
| Bâra y Oprea [[39]](#ref-39) | Mercado local | ABM + PPO | Precio / cantidad / flexibilidad | *Clearing* e interacción estratégica |

No existe un algoritmo universalmente dominante. Los elementos que aparecen de forma repetida son:

- observaciones consistentes con la información realmente disponible;
- acciones coherentes con la operativa;
- restricciones explícitas;
- reward alineado con el objetivo económico final;
- simulación suficientemente realista;
- evaluación temporal fuera de muestra;
- comparación con baselines.

# 8. Discusión: criterios metodológicos de diseño y evaluación

## 8.1. El entorno es parte del modelo

En RL, el entorno no es un mero generador de datos: define qué consecuencias puede aprender la política. Si el entorno liquida todas las operaciones al precio histórico sin modelar ejecución, la política es esencialmente *price-taker*. Esta hipótesis puede ser válida para una primera prueba de concepto si se documenta, pero no permite concluir que una estrategia precio–cantidad funcionaría igual en producción.

Una evolución razonable consiste en separar niveles de fidelidad. Un primer entorno puede validar la lógica secuencial de ajuste de posición y recompensa. Un segundo nivel puede incorporar precio límite o agresividad y una regla de casación basada en información histórica. Un tercer nivel, si el nivel de fidelidad requerido lo justifica y existen datos públicos adecuados, puede modelar libro de órdenes, impacto de mercado o interacción con otros agentes.

## 8.2. Dataset point-in-time y prevención de fuga de información

La validez de un backtest depende de reconstruir qué información estaba disponible en cada decisión. Las previsiones deben conservar su fecha de emisión; los precios de mercados posteriores no pueden aparecer en observaciones anteriores; y las transformaciones estadísticas deben ajustarse únicamente con el periodo de entrenamiento. Este principio afecta también a selección de hiperparámetros: el conjunto de test final debe permanecer aislado hasta que la política y las decisiones de diseño estén cerradas.

Un esquema recomendable es entrenamiento → validación temporal para selección → test final fuera de muestra. Para entornos no estacionarios puede añadirse evaluación *walk-forward*. La repetición con varias semillas aleatorias permite distinguir mejoras robustas de resultados dependientes de inicialización.

## 8.3. Baselines y atribución de valor

Un agente no debe evaluarse únicamente por PnL absoluto. Debe compararse, como mínimo, con: (i) una política histórica o de referencia; (ii) no operar o mantener posición; (iii) una regla heurística simple; y, cuando sea viable, (iv) una referencia de optimización con información comparable. La mejora económica debe reportarse junto con energía operada, coste de desvío, exposición, distribución de resultados y estabilidad temporal.

Cuando el agente cambia radicalmente el volumen negociado, una métrica de resultado total puede ser engañosa. Conviene añadir métricas normalizadas por energía y medidas de riesgo, además de analizar el peor desempeño por día, semana o periodo relevante.

## 8.4. Selección de algoritmo

Para una primera implementación con acciones continuas y un simulador estable, PPO, TD3 y SAC son candidatos razonables, pero responden a regímenes de datos distintos. PPO es simple y robusto en muchos entornos, aunque consume experiencia *on-policy*. TD3 y SAC reutilizan experiencia y suelen ser más eficientes en muestras. Si únicamente se dispone de un dataset fijo de transiciones y no de un simulador interactivo, CQL o IQL son opciones conceptualmente más adecuadas para offline RL [13](#ref-13), [14](#ref-14).

La selección debe realizarse mediante un protocolo común: mismo periodo de datos, mismas observaciones, mismas restricciones, misma función de recompensa, presupuesto de entrenamiento comparable y evaluación fuera de muestra. De lo contrario, atribuir la diferencia al algoritmo carece de validez experimental.

## 8.5. Reproducibilidad y gobierno del modelo

Un sistema destinado a investigación reproducible o soporte a la decisión debe registrar versión de datos, código, configuración del entorno, semilla, hiperparámetros, checkpoint y métricas. También debe permitir reconstruir una inferencia concreta: observación recibida, acción bruta, acción proyectada, motivo de cualquier restricción aplicada y resultado económico posterior.

La monitorización en producción debería diferenciar *drift* de datos, degradación de desempeño y cambio regulatorio. En mercados eléctricos, una modificación de reglas puede alterar el significado de una variable o la forma de liquidación, por lo que la actualización del entorno es tan importante como el reentrenamiento del agente.

# 9. Conclusiones

El aprendizaje por refuerzo proporciona un marco coherente para estudiar decisiones secuenciales en mercados eléctricos, pero la literatura no respalda la idea de que exista un algoritmo universalmente superior. Las aplicaciones revisadas muestran que el resultado depende de cómo se representan la información, las acciones disponibles, las restricciones y las consecuencias económicas.

Los trabajos revisados apuntan a que el diseño del entorno es tan importante como la selección del algoritmo. Elegir PPO, SAC, TD3 u otro método no compensa una representación deficiente de la información, una ejecución irreal o una recompensa mal especificada. En esas condiciones, un backtest puede ofrecer resultados atractivos sin demostrar que la política sea robusta fuera del simulador.

La evaluación debe partir de datos *point-in-time*, divisiones temporales estrictas, baselines explícitos, varias semillas cuando exista aleatoriedad y métricas que combinen resultado, riesgo y estabilidad. La complejidad del entorno debe crecer únicamente cuando haya datos y mecanismos observables que permitan justificarla.

En este contexto, RL es una herramienta útil para investigación y apoyo a la decisión cuando se acompaña de un diseño experimental reproducible y de una delimitación clara de los supuestos. El principal riesgo metodológico sigue siendo optimizar una representación incompleta del problema en lugar de aprender una política que generalice.

# Referencias

[]{#ref-1}[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[]{#ref-2}[2] Z. Zhu, Z. Hu, K. W. Chan, S. Bu, B. Zhou, and S. Xia, “Reinforcement learning in deregulated energy market: A comprehensive review,” *Applied Energy*, vol. 329, art. 120212, 2023, doi: [10.1016/j.apenergy.2022.120212](https://doi.org/10.1016/j.apenergy.2022.120212).

[]{#ref-3}[3] S. Giannelos, “Reinforcement Learning in Energy Finance: A Comprehensive Review,” *Energies*, vol. 18, no. 11, art. 2712, 2025, doi: [10.3390/en18112712](https://doi.org/10.3390/en18112712).

[]{#ref-4}[4] R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” *Journal of Risk*, vol. 2, no. 3, pp. 21–41, 2000, doi: [10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038).

[]{#ref-5}[5] Y. Chow, A. Tamar, S. Mannor, and M. Pavone, “Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach,” in *Advances in Neural Information Processing Systems 28*, 2015, pp. 1522–1530.

[]{#ref-6}[6] V. Mnih *et al.*, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, pp. 529–533, 2015, doi: [10.1038/nature14236](https://doi.org/10.1038/nature14236).

[]{#ref-7}[7] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017, doi: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347).

[]{#ref-8}[8] T. P. Lillicrap *et al.*, “Continuous control with deep reinforcement learning,” arXiv:1509.02971, 2015, doi: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971).

[]{#ref-9}[9] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 1587–1596.

[]{#ref-10}[10] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 1861–1870.

[]{#ref-11}[11] V. Mnih *et al.*, “Asynchronous Methods for Deep Reinforcement Learning,” in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 48, 2016, pp. 1928–1937.

[]{#ref-12}[12] Z. Wang *et al.*, “Dueling Network Architectures for Deep Reinforcement Learning,” in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 48, 2016, pp. 1995–2003.

[]{#ref-13}[13] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Q-Learning for Offline Reinforcement Learning,” in *Advances in Neural Information Processing Systems 33*, 2020.

[]{#ref-14}[14] I. Kostrikov, A. Nair, and S. Levine, “Offline Reinforcement Learning with Implicit Q-Learning,” in *International Conference on Learning Representations (ICLR)*, 2022.

[]{#ref-15}[15] A. Nair, A. Gupta, M. Dalal, and S. Levine, “AWAC: Accelerating Online Reinforcement Learning with Offline Datasets,” arXiv:2006.09359, 2020, doi: [10.48550/arXiv.2006.09359](https://doi.org/10.48550/arXiv.2006.09359).

[]{#ref-16}[16] R. Lowe *et al.*, “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” in *Advances in Neural Information Processing Systems 30*, 2017.

[]{#ref-17}[17] C. Yu *et al.*, “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” in *Advances in Neural Information Processing Systems 35*, 2022.

[]{#ref-18}[18] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,” in *Proc. 14th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 15, 2011, pp. 627–635.

[]{#ref-19}[19] J. Ho and S. Ermon, “Generative Adversarial Imitation Learning,” in *Advances in Neural Information Processing Systems 29*, 2016.

[]{#ref-20}[20] J. Fu, K. Luo, and S. Levine, “Learning Robust Rewards with Adversarial Inverse Reinforcement Learning,” in *International Conference on Learning Representations (ICLR)*, 2018.

[]{#ref-21}[21] Farama Foundation, “Gymnasium Documentation.” [https://gymnasium.farama.org/](https://gymnasium.farama.org/) (accessed Oct. 7, 2026).

[]{#ref-22}[22] J. K. Terry *et al.*, “PettingZoo: Gym for Multi-Agent Reinforcement Learning,” in *Advances in Neural Information Processing Systems 34, Datasets and Benchmarks Track*, 2021.

[]{#ref-23}[23] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

[]{#ref-24}[24] E. Liang *et al.*, “RLlib: Abstractions for Distributed Reinforcement Learning,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 3053–3062.

[]{#ref-25}[25] S. Huang, R. Dossa, C. Ye, and J. Braga, “CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms,” *Journal of Machine Learning Research*, vol. 23, no. 274, pp. 1–18, 2022.

[]{#ref-26}[26] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “MushroomRL: Simplifying Reinforcement Learning Research,” *Journal of Machine Learning Research*, vol. 22, no. 131, pp. 1–5, 2021.

[]{#ref-27}[27] A. Bou *et al.*, “TorchRL: A data-driven decision-making library for PyTorch,” in *International Conference on Learning Representations (ICLR)*, 2024.

[]{#ref-28}[28] J. Weng *et al.*, “Tianshou: A Highly Modularized Deep Reinforcement Learning Library,” *Journal of Machine Learning Research*, vol. 23, no. 267, pp. 1–6, 2022.

[]{#ref-29}[29] N. Harder *et al.*, “ASSUME: An agent-based simulation framework for exploring electricity market dynamics with reinforcement learning,” *SoftwareX*, vol. 30, art. 102176, 2025, doi: [10.1016/j.softx.2025.102176](https://doi.org/10.1016/j.softx.2025.102176).

[]{#ref-30}[30] T. Brown, J. Hörsch, and D. Schlachtberger, “PyPSA: Python for Power System Analysis,” *Journal of Open Research Software*, vol. 6, no. 1, art. 4, 2018, doi: [10.5334/jors.188](https://doi.org/10.5334/jors.188).

[]{#ref-31}[31] L. Thurner *et al.*, “pandapower—An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems,” *IEEE Transactions on Power Systems*, vol. 33, no. 6, pp. 6510–6521, 2018, doi: [10.1109/TPWRS.2018.2829021](https://doi.org/10.1109/TPWRS.2018.2829021).

[]{#ref-32}[32] R. D. Zimmerman, C. E. Murillo-Sánchez, and R. J. Thomas, “MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education,” *IEEE Transactions on Power Systems*, vol. 26, no. 1, pp. 12–19, 2011, doi: [10.1109/TPWRS.2010.2051168](https://doi.org/10.1109/TPWRS.2010.2051168).

[]{#ref-33}[33] OMIE, “Resultados del Mercado: mercado diario e intradiario.” [https://www.omie.es/](https://www.omie.es/) (accessed Oct. 7, 2026).

[]{#ref-34}[34] Comisión Nacional de los Mercados y la Competencia, “Resolución de 6 de marzo de 2025 por la que se modifican los procedimientos de operación eléctricos para su adaptación a la negociación cuarto-horaria en los mercados diario e intradiario,” *Boletín Oficial del Estado*, no. 65, BOE-A-2025-5342, Mar. 17, 2025.

[]{#ref-35}[35] L. Di Persio, M. Garbelli, and L. M. Giordano, “Reinforcement learning for bidding strategy optimization in day-ahead energy market,” *Energy Economics*, vol. 149, art. 108673, 2025, doi: [10.1016/j.eneco.2025.108673](https://doi.org/10.1016/j.eneco.2025.108673).

[]{#ref-36}[36] J. Li, C. Wang, Y. Zhang, and H. Wang, “Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets,” *IEEE Transactions on Energy Markets, Policy and Regulation*, vol. 2, no. 3, pp. 392–406, 2024, doi: [10.1109/TEMPR.2024.3372656](https://doi.org/10.1109/TEMPR.2024.3372656).

[]{#ref-37}[37] J. Cardo-Miota, H. Beltran, E. Pérez, S. Khadem, and M. Bahloul, “Deep reinforcement learning-based strategy for maximizing returns from renewable energy and energy storage systems in multi-electricity markets,” *Applied Energy*, vol. 388, art. 125561, 2025, doi: [10.1016/j.apenergy.2025.125561](https://doi.org/10.1016/j.apenergy.2025.125561).

[]{#ref-38}[38] A. G. Abate, X. Liu, R. Liu, and X.-B. Zhang, “Feature-driven reinforcement learning for photovoltaic in continuous intraday trading,” arXiv:2510.16021, 2025, doi: [10.48550/arXiv.2510.16021](https://doi.org/10.48550/arXiv.2510.16021).

[]{#ref-39}[39] A. Bâra and S.-V. Oprea, “Trading Strategies on Local Electricity Markets Using Agent-Based Modelling and Reinforcement Learning: Vectors to Expand Energy Communities,” *Systems Research and Behavioral Science*, vol. 43, no. 3, pp. 1188–1211, 2026, doi: [10.1002/sres.70017](https://doi.org/10.1002/sres.70017).
