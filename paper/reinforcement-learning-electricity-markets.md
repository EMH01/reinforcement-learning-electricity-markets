# Aprendizaje por refuerzo aplicado a mercados eléctricos: diseño de entornos, algoritmos, herramientas y criterios de evaluación

**Autora:** Esther M. Martin Hernández  
**Tipo:** Informe técnico de investigación  
**Versión:** 1.0 · octubre de 2026

# Resumen

Los mercados eléctricos combinan incertidumbre, restricciones operativas, información que evoluciona en el tiempo y decisiones con consecuencias económicas acumulativas. Estas características han impulsado el uso del aprendizaje por refuerzo (*reinforcement learning*, RL) como marco para estudiar problemas de decisión secuencial. Su aplicación, sin embargo, depende tanto del diseño del entorno como del algoritmo elegido: es necesario definir qué información observa el agente, qué acciones puede ejecutar, cómo se representa el objetivo, qué restricciones deben respetarse y cómo se evalúa el comportamiento fuera de muestra. Este informe reúne y organiza los principales elementos metodológicos para diseñar y analizar sistemas de RL aplicados a mercados eléctricos. Se revisan la formulación mediante procesos de decisión de Markov, los espacios de observación y acción, las funciones de recompensa, el tratamiento de restricciones, las familias de algoritmos más utilizadas y las herramientas de implementación y simulación. También se sintetizan patrones de aplicación presentes en la literatura, incluyendo estrategias de oferta, activos flexibles, generación renovable, decisiones continuas y entornos multiagente. La revisión pone especial énfasis en la prevención de fuga de información, la fidelidad del simulador, el uso de baselines, la evaluación temporal y la trazabilidad experimental.

**Palabras clave:** aprendizaje por refuerzo; mercados eléctricos; trading energético; diseño de entornos; simulación; backtesting temporal; gestión de riesgo; decisión secuencial.

# 1. Introducción

El aprendizaje por refuerzo aborda problemas en los que un agente observa información del entorno, selecciona acciones y recibe señales de recompensa con el objetivo de maximizar el retorno acumulado esperado. La formulación clásica se apoya en procesos de decisión de Markov (MDP), que proporcionan un marco para representar estados, acciones, transiciones y recompensas <a href="#ref-1">[1]</a>. En mercados eléctricos, esta estructura resulta pertinente porque las decisiones se toman bajo incertidumbre, dependen de información disponible en cada instante y pueden afectar a resultados posteriores.

La literatura ha mostrado un crecimiento sostenido de aplicaciones de RL a estrategias de oferta, despacho, almacenamiento, arbitraje, gestión de activos y control de riesgo en mercados liberalizados <a href="#ref-2">[2]</a>, <a href="#ref-3">[3]</a>. La transferencia desde un experimento académico a un sistema útil exige, no obstante, resolver varios problemas metodológicos: evitar fuga de información futura, representar con suficiente fidelidad las reglas del entorno, imponer restricciones técnicas y económicas y evaluar la política en periodos no utilizados durante el entrenamiento.

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

donde *S* es el espacio de estados, *A* el espacio de acciones, *P* la dinámica de transición, *R* la función de recompensa y *γ* el factor de descuento <a href="#ref-1">[1]</a>. En cada instante de decisión *t*, el agente recibe una observación *o_t* —que puede coincidir o no con el estado completo—, elige una acción *a_t* y recibe una recompensa *r_t*. El objetivo es aprender una política π que maximice el retorno esperado:

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

En un **espacio continuo**, la acción puede representar volumen, variación de posición, potencia de carga/descarga o parámetros de precio. DDPG, TD3 y SAC fueron desarrollados específicamente para control continuo <a href="#ref-8">[8]</a>–<a href="#ref-10">[10]</a>, mientras que PPO también se utiliza con frecuencia en espacios continuos mediante políticas estocásticas <a href="#ref-7">[7]</a>.

Una tercera alternativa es la **acción estructurada**, por ejemplo un vector de bloques precio–cantidad o una combinación de volumen y agresividad de ejecución. Este diseño es más próximo a la operativa de mercado, pero requiere un simulador capaz de traducir la acción en una probabilidad o resultado de casación. Cuando esa capa no existe, atribuir a la política una capacidad de ejecución que el entorno no modela produce una estimación optimista del rendimiento.

## 3.4. Función de recompensa

La recompensa conecta el objetivo del problema con el comportamiento que aprende la política. Una formulación genérica puede escribirse como:

**r_t = ΔJ_t − λ_risk C_risk,t − λ_op C_op,t − λ_cost C_cost,t,**

donde ΔJ representa el resultado incremental asociado al objetivo, *C_risk* un término de riesgo, *C_op* penalizaciones por acciones operativamente inviables y *C_cost* otros costes relevantes para el entorno modelado.

Los términos incluidos deben tener una interpretación económica u operativa clara. Si se incorporan medidas de riesgo de cola, el Value at Risk (VaR) resume un cuantil de pérdidas, mientras que el Conditional Value at Risk (CVaR) cuantifica la pérdida media en la cola más desfavorable. El CVaR dispone de una formulación de optimización bien establecida <a href="#ref-4">[4]</a> y ha sido utilizado en trabajos de decisión secuencial sensible al riesgo <a href="#ref-5">[5]</a>.

En algunos problemas resulta útil definir la recompensa de forma incremental respecto de una política de referencia. Este enfoque facilita atribuir la mejora al comportamiento aprendido y evita que el agente sea premiado por componentes del resultado que no dependen de sus decisiones.

## 3.5. Restricciones: penalización frente a proyección

Las restricciones pueden tratarse de tres maneras. La primera es penalizar violaciones dentro de la recompensa. La segunda es **recortar o proyectar** la acción a la región factible antes de ejecutarla. La tercera es diseñar el espacio de acciones de modo que solo genere decisiones válidas. En aplicaciones industriales, las dos últimas opciones son preferibles para restricciones duras —límites de potencia, energía, posición o mercado—, mientras que las penalizaciones son apropiadas para preferencias blandas como reducir rotación, exposición o riesgo.

Debe distinguirse entre la acción propuesta por la política y la acción finalmente ejecutada por el entorno. Esta trazabilidad es necesaria para auditar por qué una política obtiene un resultado determinado y para evitar que el agente aprenda a explotar errores de modelado.

# 4. Algoritmos y arquitecturas relevantes

## 4.1. Métodos basados en valor, política y actor–critic

Los métodos **basados en valor** estiman funciones como *V(s)* o *Q(s,a)* y derivan la decisión a partir de esas estimaciones. DQN demostró la capacidad de combinar redes neuronales y Q-learning en espacios de acción discretos <a href="#ref-6">[6]</a>. Las arquitecturas *dueling* separan el valor del estado y la ventaja relativa de las acciones, lo que puede mejorar la estimación cuando varias acciones tienen efectos similares <a href="#ref-12">[12]</a>.

Los métodos **basados en política** optimizan directamente una distribución de acciones. PPO limita el tamaño efectivo de las actualizaciones mediante un objetivo sustituto con *clipping*, buscando un equilibrio entre estabilidad y simplicidad <a href="#ref-7">[7]</a>.

Los métodos **actor–critic** combinan una política (actor) y una estimación de valor (critic). DDPG extiende ideas de gradiente de política determinista a acciones continuas <a href="#ref-8">[8]</a>. TD3 introduce doble crítico, objetivos suavizados y actualizaciones retrasadas para reducir sesgos de sobreestimación <a href="#ref-9">[9]</a>. SAC adopta un objetivo de máxima entropía y aprendizaje *off-policy*, favoreciendo exploración y reutilización de experiencia <a href="#ref-10">[10]</a>. A2C es una variante síncrona de la familia actor–critic asociada al trabajo que popularizó A3C; no debe confundirse con “Asynchronous Advantage Actor-Critic”, denominación que corresponde a A3C <a href="#ref-11">[11]</a>.

## 4.2. Online RL y offline RL

En **online RL**, la política genera nueva experiencia al interactuar con un entorno real o simulado. PPO es *on-policy*: aprende principalmente con trayectorias producidas por la política actual o una versión muy próxima <a href="#ref-7">[7]</a>. SAC y TD3 son *off-policy* y pueden reutilizar transiciones almacenadas en un *replay buffer* <a href="#ref-9">[9]</a>, <a href="#ref-10">[10]</a>.

En **offline RL**, el aprendizaje se realiza a partir de un dataset estático sin interacción adicional durante el entrenamiento. La dificultad principal es el *distribution shift*: la política puede proponer acciones poco representadas en el dataset y el estimador de valor puede asignarles valores excesivamente optimistas. CQL aborda este problema mediante estimaciones conservadoras de Q <a href="#ref-13">[13]</a>, mientras que IQL evita evaluar explícitamente acciones fuera de la distribución mediante una formulación implícita <a href="#ref-14">[14]</a>. AWAC es especialmente relevante cuando se desea aprovechar datos previos y continuar posteriormente con ajuste online <a href="#ref-15">[15]</a>.

Es importante no confundir **offline RL** con entrenar un algoritmo online dentro de un simulador alimentado por datos históricos. En el segundo caso, el agente sí interactúa y genera transiciones en un entorno simulado; que las series exógenas procedan del histórico no convierte automáticamente el método en offline RL.

## 4.3. Arquitecturas neuronales y memoria temporal

Cuando el estado es un vector tabular de variables, una MLP constituye un baseline razonable por su simplicidad. Las redes recurrentes pueden aportar memoria cuando la observación es parcial o existe dependencia temporal no resumida por el estado. Los Transformers permiten modelar relaciones de largo alcance y se han incorporado a sistemas de trading y almacenamiento, pero su uso debe justificarse mediante comparación experimental frente a arquitecturas más sencillas.

Las CNN son adecuadas cuando la entrada posee estructura espacial o matricial; por ejemplo, mapas o representaciones densas del libro de órdenes. Los autoencoders pueden emplearse como reducción de dimensionalidad, pero añaden otra etapa de aprendizaje cuyo beneficio debe demostrarse.

## 4.4. Aprendizaje multiagente y jerárquico

En un entorno con participantes estratégicos, el problema puede formularse como aprendizaje multiagente. MADDPG introdujo un esquema de entrenamiento centralizado y ejecución descentralizada para entornos cooperativos y competitivos <a href="#ref-16">[16]</a>. MAPPO mostró que variantes de PPO pueden constituir baselines muy competitivos en tareas cooperativas <a href="#ref-17">[17]</a>.

El **modelado basado en agentes (ABM)** y el **aprendizaje multiagente (MARL)** están relacionados, pero no son sinónimos. ABM es un paradigma de simulación de sistemas formados por entidades autónomas con reglas de comportamiento; MARL se refiere específicamente a agentes que aprenden políticas mediante RL. Un ABM puede contener agentes no aprendices, y un sistema MARL no necesita adoptar toda la metodología típica de ABM.

El aprendizaje jerárquico introduce decisiones en diferentes escalas temporales —por ejemplo, una política de alto nivel que selecciona un modo operativo y otra de bajo nivel que decide volúmenes—. Aunque resulta atractivo para problemas complejos, incrementa la dificultad de entrenamiento y de atribución causal del rendimiento, por lo que conviene utilizarlo cuando exista una jerarquía operacional real.

## 4.5. Aprendizaje por imitación y regularización con el histórico

El histórico operativo puede utilizarse para algo más que comparar resultados. El *behavioral cloning* aprende una política supervisada a partir de pares estado–acción, proporcionando una inicialización plausible, aunque por sí solo reproduce el comportamiento observado y no optimiza explícitamente una recompensa.

DAgger reduce el error de distribución de la imitación supervisada agregando de manera iterativa estados visitados por la política y consultando la acción de un experto <a href="#ref-18">[18]</a>. GAIL aprende una política cuya distribución de trayectorias se aproxima a la de las demostraciones mediante un esquema adversarial <a href="#ref-19">[19]</a>, mientras que AIRL busca recuperar una función de recompensa más interpretable y transferible <a href="#ref-20">[20]</a>. En trading energético, estas técnicas son útiles principalmente como inicialización, restricción de soporte o benchmark; no sustituyen la evaluación económica fuera de muestra.

# 5. Implementación técnica y herramientas

## 5.1. Interfaces de entorno

La separación entre agente y entorno es crítica para la reproducibilidad. Gymnasium mantiene una API ampliamente utilizada para entornos de RL y formaliza métodos como `reset()` y `step()` <a href="#ref-21">[21]</a>. Para problemas multiagente, PettingZoo define interfaces específicas para interacción simultánea o secuencial entre múltiples agentes <a href="#ref-22">[22]</a>.

Esta separación permite cambiar el algoritmo sin reescribir la lógica de mercado, siempre que se mantengan estables las observaciones, acciones, transiciones y recompensas.

## 5.2. Frameworks de aprendizaje

**Tabla 1. Frameworks relevantes para prototipado e investigación**

| Framework | Uso principal | Observación |
|---|---|---|
| Stable-Baselines3 | Baselines model-free (PPO, SAC, TD3, DQN, A2C) | Adecuado para prototipado reproducible con PyTorch <a href="#ref-23">[23]</a>. |
| RLlib | Entrenamiento distribuido y multiagente | Proporciona abstracciones para escalar experimentos y políticas <a href="#ref-24">[24]</a>. |
| CleanRL | Implementaciones compactas | Útil para auditoría del algoritmo y experimentos controlados <a href="#ref-25">[25]</a>. |
| MushroomRL | Investigación modular | Facilita experimentar con familias model-free y model-based <a href="#ref-26">[26]</a>. |
| TorchRL | Componentes RL dentro de PyTorch | Apropiado para pipelines personalizados, datasets y objetivos de RL <a href="#ref-27">[27]</a>. |
| Tianshou | Colección modular de algoritmos | Ofrece componentes reutilizables y soporte para múltiples paradigmas <a href="#ref-28">[28]</a>. |

En la práctica, la elección del framework depende de la reproducibilidad que se necesite, los algoritmos disponibles, la trazabilidad de los experimentos y la facilidad de integración con el resto del sistema.

## 5.3. Simulación de mercado, activos y red

Un entorno de RL para mercados eléctricos puede separarse conceptualmente en varias capas: **mercado**, **activo o estado interno**, **ejecución y resultado económico** y **datos exógenos**. La capa de mercado define las reglas relevantes para la decisión; la capa de activo aplica restricciones físicas cuando existen; la ejecución traduce la acción del agente en una consecuencia observable; y los datos exógenos aportan la información disponible en cada instante.

ASSUME ofrece un marco abierto de simulación basada en agentes para mercados eléctricos e integra estrategias con aprendizaje por refuerzo <a href="#ref-29">[29]</a>. Para modelado físico, PyPSA permite análisis y optimización de sistemas energéticos <a href="#ref-30">[30]</a>, pandapower facilita flujo de potencia y otras funciones de análisis de redes eléctricas <a href="#ref-31">[31]</a>, y MATPOWER continúa siendo una referencia abierta para estudios de flujo de potencia y *optimal power flow* <a href="#ref-32">[32]</a>. Estas herramientas no son intercambiables: su utilidad depende de si el problema requiere representar reglas de mercado, restricciones físicas, interacción entre agentes o una combinación de estas capas.

Un simulador útil no necesita reproducir todo el sistema eléctrico. Debe incluir únicamente los mecanismos que afectan a las decisiones analizadas y documentar de forma explícita las simplificaciones adoptadas. Añadir componentes que no modifican el comportamiento del agente incrementa la complejidad sin mejorar necesariamente la validez del experimento.

# 6. Datos, reglas de mercado y validez del entorno

La calidad de un experimento de RL depende de que las observaciones reflejen la información disponible en el momento de la decisión. Los mercados eléctricos ofrecen distintas fuentes de datos públicos —por ejemplo, precios, resultados agregados, variables de demanda, generación o meteorología—, pero la cobertura y el nivel de detalle varían entre mercados y periodos. La existencia de información agregada no implica disponer de órdenes individuales, estrategias de participantes ni condiciones de ejecución completas.

La resolución temporal del entorno debe responder al fenómeno que se desea estudiar y a la información realmente disponible. Elegir una granularidad más fina no mejora por sí sola el modelo si los datos o las reglas no permiten reconstruir de forma fiable las transiciones entre decisiones. Del mismo modo, cualquier transformación, previsión o variable derivada debe respetar el principio *point-in-time*.

Las reglas de mercado también forman parte del entorno. Cuando cambian, pueden modificar el significado de una observación, una acción o una función de coste. Por ello, la versión de las reglas, los datos y la configuración del simulador deben registrarse junto con cada experimento <a href="#ref-2">[2]</a>, <a href="#ref-3">[3]</a>.

# 7. Casos publicados en mercados eléctricos y trading

Los siguientes trabajos se incluyen porque representan formulaciones relevantes dentro de la literatura reciente: estrategias de oferta, acciones continuas, operación multi-mercado, almacenamiento, generación renovable y simulación multiagente. Cada caso se resume de forma independiente y únicamente en los términos descritos por sus autores; la selección no implica una arquitectura conjunta ni una propuesta de implementación concreta.

## 7.1. Curvas de oferta precio-cantidad con DDPG

**Referencia principal:** Di Persio, Garbelli y Giordano, *Energy Economics* (2025) <a href="#ref-33">[33]</a>.

**Objetivo.** El estudio plantea la generación de curvas de oferta escalonadas precio-cantidad para un participante en el mercado *day-ahead*, con el objetivo de maximizar el beneficio bajo restricciones de producción.

**Enfoque.** Los autores utilizan DDPG, un método actor-critic de acción continua. La elección responde a que la acción no es una decisión binaria, sino un vector multidimensional que representa una curva de oferta completa <a href="#ref-33">[33]</a>.

**Estado e información de entrada.** El modelo utiliza información histórica de precios del mercado eléctrico italiano junto con parámetros relacionados con costes y capacidad de producción.

**Acción.** La política genera una curva de oferta escalonada precio-cantidad.

**Restricciones.** La acción debe respetar los límites de producción y la estructura admitida para la oferta.

**Reward.** La recompensa se deriva del beneficio generado por la estrategia. El resultado se normaliza respecto a un máximo teórico definido bajo las hipótesis del experimento.

**Evaluación y resultados.** Los autores reportan valores de hasta aproximadamente un 85 % de recompensa normalizada en los experimentos presentados <a href="#ref-33">[33]</a>.

**Conclusión metodológica.** El trabajo demuestra que una acción precio-cantidad estructurada puede abordarse mediante control continuo. También pone de relieve que la validez del resultado depende de representar explícitamente el mecanismo que transforma una oferta en energía ejecutada y resultado económico.

## 7.2. BESS multi-mercado y representación temporal

**Referencia principal:** Li, Wang, Zhang y Wang, *IEEE Transactions on Energy Markets, Policy and Regulation* (2024) <a href="#ref-34">[34]</a>.

**Objetivo.** El trabajo estudia la participación conjunta de un sistema de almacenamiento en batería (BESS) en el mercado *spot* y en servicios australianos de reserva de contingencia.

**Arquitectura.** El método TempDRL combina un extractor temporal basado en Transformer con un agente Soft Actor-Critic (SAC) para decisiones continuas <a href="#ref-34">[34]</a>.

**Estado.** La observación incorpora precios de varios mercados, una representación temporal aprendida y el estado de carga del BESS.

**Acción.** El agente toma decisiones continuas de *bidding* y asignación de capacidad entre los distintos servicios considerados.

**Restricciones.** El entorno incorpora límites de capacidad y potencia, además de la compatibilidad entre la energía comprometida en diferentes servicios.

**Reward.** La recompensa agrega los ingresos procedentes de las distintas vías de mercado e incorpora las consecuencias operativas de las decisiones sobre el almacenamiento.

**Datos y evaluación.** El estudio emplea precios del Australian National Electricity Market y compara el método con estrategias basadas en optimización y con otros enfoques DRL. Los resultados reportados muestran ventajas de la participación conjunta frente al tratamiento aislado de los mercados en los escenarios estudiados <a href="#ref-34">[34]</a>.

**Conclusión metodológica.** En problemas con almacenamiento, el estado interno del activo condiciona las decisiones futuras y genera costes de oportunidad entre servicios. Las arquitecturas temporales pueden aportar valor, pero su beneficio debe comprobarse frente a modelos más sencillos.

## 7.3. Renovables con almacenamiento en varios mercados

**Referencia principal:** Cardo-Miota *et al.*, *Applied Energy* (2025) <a href="#ref-35">[35]</a>.

**Objetivo.** El estudio busca maximizar los ingresos de un sistema renovable con batería mediante participación simultánea en un mercado *day-ahead* y en servicios de reserva del programa DS3 irlandés.

**Enfoque.** El problema se formula como un MDP y se utiliza TD3 como algoritmo actor-critic para acción continua <a href="#ref-35">[35]</a>.

**Estado.** La observación combina señales de mercado, producción renovable y estado del almacenamiento.

**Datos.** El caso de estudio utiliza datos históricos de una planta fotovoltaica de 7 MW y un BESS de 1 MWh <a href="#ref-35">[35]</a>.

**Acción.** La política toma decisiones continuas sobre el uso del almacenamiento y la participación en los servicios considerados.

**Restricciones.** Se modelan capacidad, potencia, estado de carga y compatibilidad con los servicios de mercado.

**Reward.** La recompensa representa el resultado económico agregado de la operación, sujeto a las restricciones físicas del activo.

**Evaluación.** Los autores comparan el agente con estrategias de referencia y reportan beneficios económicos adicionales en los escenarios evaluados <a href="#ref-35">[35]</a>.

**Conclusión metodológica.** Las restricciones físicas deben formar parte del entorno y de la dinámica de transición. En problemas con varias fuentes de ingreso, desagregar los resultados por componente ayuda a determinar de dónde procede una mejora observada.

## 7.4. Trading continuo de generación fotovoltaica bajo incertidumbre

**Referencia principal:** Abate, Liu, Liu y Zhang (2025) <a href="#ref-36">[36]</a>.

**Objetivo.** El estudio analiza trading intradiario continuo para una planta fotovoltaica bajo incertidumbre de generación y precios, considerando conjuntamente el resultado de trading y la exposición a desbalances.

**Algoritmo.** Los autores utilizan PPO con una política diseñada para conservar interpretabilidad <a href="#ref-36">[36]</a>.

**Estado.** La observación combina variables de mercado, información histórica, señales de microestructura y variables relacionadas con la producción y sus previsiones.

**Acción.** La estrategia se ajusta secuencialmente a medida que cambia la información disponible.

**Reward.** La recompensa combina el resultado del trading con una penalización asociada al desbalance, evitando que el agente optimice únicamente el beneficio bruto de las operaciones.

**Evaluación.** El entrenamiento utiliza datos históricos y la evaluación se realiza fuera de muestra. El trabajo reporta mejoras frente a los benchmarks considerados en los escenarios analizados <a href="#ref-36">[36]</a>.

**Conclusión metodológica.** El caso muestra la importancia de que la recompensa represente todas las consecuencias económicas relevantes de una decisión y de mantener una separación temporal estricta entre entrenamiento y evaluación. La interpretabilidad de la política puede facilitar además el análisis posterior de su comportamiento.

## 7.5. Simulación multiagente y mercados locales

**Referencia principal:** Bâra y Oprea, *Systems Research and Behavioral Science* (2026) <a href="#ref-37">[37]</a>.

**Objetivo.** El trabajo modela trading en mercados eléctricos locales con prosumidores heterogéneos e integra generación, consumo flexible, vehículos eléctricos y calefacción.

**Arquitectura.** La simulación combina Mesa para el modelado basado en agentes, PyMarket para el mecanismo de mercado y PPO de Stable-Baselines3 como política de *bidding* <a href="#ref-37">[37]</a>. Esta configuración separa la simulación del ecosistema de agentes de la optimización de las decisiones mediante RL.

**Estrategias comparadas.** En una estrategia el agente determina principalmente el precio de puja; en otra optimiza precio y cantidad y coordina flexibilidad de cargas <a href="#ref-37">[37]</a>.

**Datos y simulación.** El estudio utiliza una comunidad de 114 hogares, datos de consumo a resolución de 15 minutos y una simulación anual con varios mecanismos de mercado local <a href="#ref-37">[37]</a>.

**Reward.** La recompensa se calcula a partir del resultado obtenido después del *clearing* de mercado y guía la política hacia mejores decisiones de *bidding*.

**Métricas y resultados.** Los autores emplean indicadores de comunidad y ahorro financiero. En los escenarios reportados, la estrategia que incorpora precio, cantidad y flexibilidad supera a la alternativa más simple; por ejemplo, el ahorro financiero pasa del 11,58 % al 17,88 % en verano y del 4,21 % al 5,38 % en invierno <a href="#ref-37">[37]</a>.

**Conclusión metodológica.** El estudio es útil para analizar cómo la interacción entre participantes, el mecanismo de *clearing* y la flexibilidad modifican el entorno de decisión. Sus resultados deben interpretarse dentro de las reglas y la escala específicas del mercado simulado.

## 7.6. Síntesis comparativa de los casos

**Tabla 2. Casos publicados y elementos metodológicos**

| Estudio | Contexto publicado | Algoritmo | Acción | Elemento metodológico |
|---|---|---|---|---|
| Di Persio *et al.* <a href="#ref-33">[33]</a> | *Day-ahead*, generación | DDPG | Curva precio-cantidad | Acción continua estructurada y modelado de ejecución |
| Li *et al.* <a href="#ref-34">[34]</a> | BESS en varios servicios | SAC + Transformer | Continua multi-mercado | Estado temporal, SoC y coste de oportunidad |
| Cardo-Miota *et al.* <a href="#ref-35">[35]</a> | Renovable + BESS | TD3 | Operación continua | Restricciones físicas dentro del entorno |
| Abate *et al.* <a href="#ref-36">[36]</a> | PV y trading continuo | PPO | Ajuste secuencial | Reward completo y evaluación fuera de muestra |
| Bâra y Oprea <a href="#ref-37">[37]</a> | Mercado local | ABM + PPO | Precio, cantidad y flexibilidad | Clearing e interacción estratégica |

Los cinco estudios utilizan formulaciones diferentes y no permiten identificar un algoritmo universalmente dominante. Sí comparten varios requisitos metodológicos: observaciones consistentes con la información realmente disponible, acciones coherentes con el mecanismo representado, restricciones explícitas, una recompensa alineada con el objetivo evaluado, simulación suficiente para interpretar la acción y validación fuera de muestra frente a referencias adecuadas.

# 8. Discusión: criterios metodológicos de diseño y evaluación

## 8.1. El entorno es parte del modelo

En RL, el entorno no es un mero generador de datos: define qué consecuencias puede aprender la política. Si una simulación asigna directamente resultados históricos sin modelar los mecanismos que conectan una acción con su ejecución, se está introduciendo una hipótesis simplificadora. Esa aproximación puede ser útil en una fase exploratoria, siempre que se documente y que las conclusiones se limiten al alcance que permite el simulador.

La fidelidad debe aumentar solo cuando el objetivo de la investigación y la disponibilidad de datos lo justifican. Un entorno más complejo no es necesariamente más válido; lo importante es que cada componente añadido represente un mecanismo relevante y verificable.

## 8.2. Dataset point-in-time y prevención de fuga de información

La validez de un backtest depende de reconstruir qué información estaba disponible en cada decisión. Las previsiones deben conservar su fecha de emisión; las variables futuras no pueden aparecer en observaciones anteriores; y las transformaciones estadísticas deben ajustarse únicamente con el periodo de entrenamiento. Este principio afecta también a selección de hiperparámetros: el conjunto de test final debe permanecer aislado hasta que la política y las decisiones de diseño estén cerradas.

Un esquema recomendable es entrenamiento → validación temporal para selección → test final fuera de muestra. Para entornos no estacionarios puede añadirse evaluación *walk-forward*. La repetición con varias semillas aleatorias permite distinguir mejoras robustas de resultados dependientes de inicialización.

## 8.3. Baselines y atribución de valor

Un agente no debe evaluarse únicamente por un resultado económico agregado. Debe compararse, como mínimo, con una política de referencia, una regla heurística simple y, cuando sea viable, una referencia de optimización con información comparable. La mejora debe reportarse junto con medidas de exposición, costes relevantes, distribución de resultados y estabilidad temporal.

Cuando el agente cambia radicalmente el volumen negociado, una métrica de resultado total puede ser engañosa. Conviene añadir métricas normalizadas por energía y medidas de riesgo, además de analizar el peor desempeño por día, semana o periodo relevante.

## 8.4. Selección de algoritmo

Para una primera implementación con acciones continuas y un simulador estable, PPO, TD3 y SAC son candidatos razonables, pero responden a regímenes de datos distintos. PPO es simple y robusto en muchos entornos, aunque consume experiencia *on-policy*. TD3 y SAC reutilizan experiencia y suelen ser más eficientes en muestras. Si únicamente se dispone de un dataset fijo de transiciones y no de un simulador interactivo, CQL o IQL son opciones conceptualmente más adecuadas para offline RL <a href="#ref-13">[13]</a>, <a href="#ref-14">[14]</a>.

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

<a id="ref-1"></a>[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

<a id="ref-2"></a>[2] Z. Zhu, Z. Hu, K. W. Chan, S. Bu, B. Zhou, and S. Xia, “Reinforcement learning in deregulated energy market: A comprehensive review,” *Applied Energy*, vol. 329, art. 120212, 2023, doi: [10.1016/j.apenergy.2022.120212](https://doi.org/10.1016/j.apenergy.2022.120212).

<a id="ref-3"></a>[3] S. Giannelos, “Reinforcement Learning in Energy Finance: A Comprehensive Review,” *Energies*, vol. 18, no. 11, art. 2712, 2025, doi: [10.3390/en18112712](https://doi.org/10.3390/en18112712).

<a id="ref-4"></a>[4] R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” *Journal of Risk*, vol. 2, no. 3, pp. 21–41, 2000, doi: [10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038).

<a id="ref-5"></a>[5] Y. Chow, A. Tamar, S. Mannor, and M. Pavone, “Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach,” in *Advances in Neural Information Processing Systems 28*, 2015, pp. 1522–1530.

<a id="ref-6"></a>[6] V. Mnih *et al.*, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, pp. 529–533, 2015, doi: [10.1038/nature14236](https://doi.org/10.1038/nature14236).

<a id="ref-7"></a>[7] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017, doi: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347).

<a id="ref-8"></a>[8] T. P. Lillicrap *et al.*, “Continuous control with deep reinforcement learning,” arXiv:1509.02971, 2015, doi: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971).

<a id="ref-9"></a>[9] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 1587–1596.

<a id="ref-10"></a>[10] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 1861–1870.

<a id="ref-11"></a>[11] V. Mnih *et al.*, “Asynchronous Methods for Deep Reinforcement Learning,” in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 48, 2016, pp. 1928–1937.

<a id="ref-12"></a>[12] Z. Wang *et al.*, “Dueling Network Architectures for Deep Reinforcement Learning,” in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 48, 2016, pp. 1995–2003.

<a id="ref-13"></a>[13] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Q-Learning for Offline Reinforcement Learning,” in *Advances in Neural Information Processing Systems 33*, 2020.

<a id="ref-14"></a>[14] I. Kostrikov, A. Nair, and S. Levine, “Offline Reinforcement Learning with Implicit Q-Learning,” in *International Conference on Learning Representations (ICLR)*, 2022.

<a id="ref-15"></a>[15] A. Nair, A. Gupta, M. Dalal, and S. Levine, “AWAC: Accelerating Online Reinforcement Learning with Offline Datasets,” arXiv:2006.09359, 2020, doi: [10.48550/arXiv.2006.09359](https://doi.org/10.48550/arXiv.2006.09359).

<a id="ref-16"></a>[16] R. Lowe *et al.*, “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” in *Advances in Neural Information Processing Systems 30*, 2017.

<a id="ref-17"></a>[17] C. Yu *et al.*, “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” in *Advances in Neural Information Processing Systems 35*, 2022.

<a id="ref-18"></a>[18] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,” in *Proc. 14th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 15, 2011, pp. 627–635.

<a id="ref-19"></a>[19] J. Ho and S. Ermon, “Generative Adversarial Imitation Learning,” in *Advances in Neural Information Processing Systems 29*, 2016.

<a id="ref-20"></a>[20] J. Fu, K. Luo, and S. Levine, “Learning Robust Rewards with Adversarial Inverse Reinforcement Learning,” in *International Conference on Learning Representations (ICLR)*, 2018.

<a id="ref-21"></a>[21] Farama Foundation, “Gymnasium Documentation.” [https://gymnasium.farama.org/](https://gymnasium.farama.org/) (accessed Oct. 7, 2026).

<a id="ref-22"></a>[22] J. K. Terry *et al.*, “PettingZoo: Gym for Multi-Agent Reinforcement Learning,” in *Advances in Neural Information Processing Systems 34, Datasets and Benchmarks Track*, 2021.

<a id="ref-23"></a>[23] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

<a id="ref-24"></a>[24] E. Liang *et al.*, “RLlib: Abstractions for Distributed Reinforcement Learning,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 3053–3062.

<a id="ref-25"></a>[25] S. Huang, R. Dossa, C. Ye, and J. Braga, “CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms,” *Journal of Machine Learning Research*, vol. 23, no. 274, pp. 1–18, 2022.

<a id="ref-26"></a>[26] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “MushroomRL: Simplifying Reinforcement Learning Research,” *Journal of Machine Learning Research*, vol. 22, no. 131, pp. 1–5, 2021.

<a id="ref-27"></a>[27] A. Bou *et al.*, “TorchRL: A data-driven decision-making library for PyTorch,” in *International Conference on Learning Representations (ICLR)*, 2024.

<a id="ref-28"></a>[28] J. Weng *et al.*, “Tianshou: A Highly Modularized Deep Reinforcement Learning Library,” *Journal of Machine Learning Research*, vol. 23, no. 267, pp. 1–6, 2022.

<a id="ref-29"></a>[29] N. Harder *et al.*, “ASSUME: An agent-based simulation framework for exploring electricity market dynamics with reinforcement learning,” *SoftwareX*, vol. 30, art. 102176, 2025, doi: [10.1016/j.softx.2025.102176](https://doi.org/10.1016/j.softx.2025.102176).

<a id="ref-30"></a>[30] T. Brown, J. Hörsch, and D. Schlachtberger, “PyPSA: Python for Power System Analysis,” *Journal of Open Research Software*, vol. 6, no. 1, art. 4, 2018, doi: [10.5334/jors.188](https://doi.org/10.5334/jors.188).

<a id="ref-31"></a>[31] L. Thurner *et al.*, “pandapower—An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems,” *IEEE Transactions on Power Systems*, vol. 33, no. 6, pp. 6510–6521, 2018, doi: [10.1109/TPWRS.2018.2829021](https://doi.org/10.1109/TPWRS.2018.2829021).

<a id="ref-32"></a>[32] R. D. Zimmerman, C. E. Murillo-Sánchez, and R. J. Thomas, “MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education,” *IEEE Transactions on Power Systems*, vol. 26, no. 1, pp. 12–19, 2011, doi: [10.1109/TPWRS.2010.2051168](https://doi.org/10.1109/TPWRS.2010.2051168).

<a id="ref-33"></a>[33] L. Di Persio, M. Garbelli, and L. M. Giordano, “Reinforcement learning for bidding strategy optimization in day-ahead energy market,” *Energy Economics*, vol. 149, art. 108673, 2025, doi: [10.1016/j.eneco.2025.108673](https://doi.org/10.1016/j.eneco.2025.108673).

<a id="ref-34"></a>[34] J. Li, C. Wang, Y. Zhang, and H. Wang, “Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets,” *IEEE Transactions on Energy Markets, Policy and Regulation*, vol. 2, no. 3, pp. 392–406, 2024, doi: [10.1109/TEMPR.2024.3372656](https://doi.org/10.1109/TEMPR.2024.3372656).

<a id="ref-35"></a>[35] J. Cardo-Miota, H. Beltran, E. Pérez, S. Khadem, and M. Bahloul, “Deep reinforcement learning-based strategy for maximizing returns from renewable energy and energy storage systems in multi-electricity markets,” *Applied Energy*, vol. 388, art. 125561, 2025, doi: [10.1016/j.apenergy.2025.125561](https://doi.org/10.1016/j.apenergy.2025.125561).

<a id="ref-36"></a>[36] A. G. Abate, X. Liu, R. Liu, and X.-B. Zhang, “Feature-driven reinforcement learning for photovoltaic in continuous intraday trading,” arXiv:2510.16021, 2025, doi: [10.48550/arXiv.2510.16021](https://doi.org/10.48550/arXiv.2510.16021).

<a id="ref-37"></a>[37] A. Bâra and S.-V. Oprea, “Trading Strategies on Local Electricity Markets Using Agent-Based Modelling and Reinforcement Learning: Vectors to Expand Energy Communities,” *Systems Research and Behavioral Science*, vol. 43, no. 3, pp. 1188–1211, 2026, doi: [10.1002/sres.70017](https://doi.org/10.1002/sres.70017).
