# Aprendizaje por refuerzo para trading en mercados eléctricos: diseño del entorno, algoritmos, herramientas y evidencia aplicada

**Autora:** Esther M. Martin Hernández  
**Estado:** Revisión técnica de investigación completada · Versión 1.0 · octubre de 2026  
**Revisión por pares:** No

## Resumen

La creciente penetración de generación renovable, la volatilidad de los precios y la sucesión de mercados con distintos horizontes temporales convierten la operación eléctrica en un problema de decisión secuencial bajo incertidumbre. El aprendizaje por refuerzo (*reinforcement learning*, RL) ofrece un marco formal para aprender políticas de decisión a partir de la interacción con un entorno. En trading eléctrico, sin embargo, su utilidad depende menos de elegir un algoritmo concreto que de representar bien el mercado, limitar la información a la disponible en cada instante, definir una recompensa con sentido económico y evaluar fuera de muestra. Este informe revisa los elementos necesarios para diseñar sistemas de RL aplicados a mercados eléctricos: formulación mediante procesos de decisión de Markov, espacios de estado y acción, recompensas y restricciones, principales familias de algoritmos —PPO, DDPG, TD3, SAC y métodos de RL offline— y herramientas de implementación y simulación. Se examinan además cinco estudios recientes sobre curvas de oferta, almacenamiento multi-mercado, generación renovable con baterías, trading intradiario continuo y mercados locales multiagente. El análisis se completa con las particularidades del mercado ibérico, donde la granularidad cuarto-horaria, la liquidación de desvíos y la disponibilidad de información pública condicionan tanto el estado observable como la recompensa. La literatura revisada muestra que RL puede ser útil en este dominio, pero también que un backtesting temporal riguroso, baselines sólidos, control de riesgo y validación de la factibilidad operativa son condiciones necesarias para interpretar los resultados.

**Palabras clave:** aprendizaje por refuerzo; mercados eléctricos; trading energético; PPO; actor–critic; desvíos; mercado intradiario; backtesting.

## 1. Introducción

El aprendizaje por refuerzo aborda problemas en los que un agente observa información del entorno, selecciona acciones y recibe señales de recompensa con el objetivo de maximizar el retorno acumulado esperado. La formulación clásica se apoya en procesos de decisión de Markov (MDP), que proporcionan un marco para representar estados, acciones, transiciones y recompensas <a href="#ref-1">[1]</a>. En mercados eléctricos, esta estructura resulta especialmente pertinente porque las decisiones de oferta, compra, venta o ajuste de posición se encadenan en el tiempo, están condicionadas por nueva información y generan consecuencias económicas que pueden materializarse en mercados posteriores o en la liquidación de desvíos.

La literatura ha mostrado un crecimiento sostenido de aplicaciones de RL a estrategias de oferta, despacho, almacenamiento, arbitraje y gestión de riesgo en mercados liberalizados <a href="#ref-2">[2]</a>, <a href="#ref-3">[3]</a>. Sin embargo, la transferencia desde un experimento académico a una herramienta operativa exige resolver varios problemas metodológicos: evitar fuga de información futura, representar de forma realista las reglas de mercado, imponer límites técnicos y comerciales, modelar costes de desbalance y evaluar la política en periodos no utilizados durante el entrenamiento.

Este informe organiza esos elementos con un objetivo práctico: disponer de un marco para diseñar y evaluar un sistema de trading energético basado en RL. Para ello, cubre cuatro frentes: (i) formalizar el entorno de decisión y sus componentes; (ii) ordenar las familias de algoritmos y arquitecturas según el tipo de problema; (iii) identificar herramientas de implementación y simulación pertinentes; y (iv) sintetizar evidencia aplicada reciente y discutir su transferibilidad al mercado eléctrico ibérico.

## 2. Alcance y naturaleza de la revisión

El documento se plantea como una **revisión narrativa y técnica orientada al diseño**, no como una revisión sistemática. Se priorizan trabajos fundacionales de RL, artículos revisados por pares, publicaciones de software científico y documentación oficial de operadores o reguladores. Los ejemplos aplicados se seleccionan por su cercanía a problemas de trading eléctrico: construcción de ofertas, acciones continuas, operación multi-mercado, almacenamiento, desvíos, intradiario continuo y entornos multiagente.

Por tanto, las conclusiones deben leerse como una síntesis técnica, no como un metaanálisis exhaustivo de toda la literatura. Cada fuente se utiliza para respaldar la afirmación concreta a la que se refiere, sin extrapolar resultados de un entorno experimental a otro cuando no hay base para hacerlo.

### 2.1. Fuentes y alcance

El informe se ha elaborado a partir de literatura científica, documentación oficial y fuentes públicas.

## 3. Fundamentos y diseño del entorno de aprendizaje por refuerzo

### 3.1. Formulación como proceso de decisión secuencial

Un MDP puede expresarse como la tupla

**M = (S, A, P, R, γ),**

donde *S* es el espacio de estados, *A* el espacio de acciones, *P* la dinámica de transición, *R* la función de recompensa y *γ* el factor de descuento <a href="#ref-1">[1]</a>. En cada instante de decisión *t*, el agente recibe una observación *o_t* —que puede coincidir o no con el estado completo—, elige una acción *a_t* y recibe una recompensa *r_t*. El objetivo es aprender una política π que maximice el retorno esperado:

**G_t = Σ(k=0…∞) γ^k r_(t+k+1).**

En trading eléctrico, la hipótesis de observabilidad completa rara vez es estrictamente válida: el agente desconoce el futuro y puede no observar el libro de órdenes completo, la estrategia de otros participantes o determinadas restricciones no observables o no publicadas. Por ello, es más preciso hablar de **observación** cuando el vector entregado al agente representa solo la información disponible en ese instante.

### 3.2. Espacio de estado u observación

El estado operativo puede organizarse en cuatro bloques.

**a) Contexto de mercado y variables exógenas.** Incluye precios observados, diferenciales entre mercados, profundidad o liquidez cuando esté disponible, previsiones de generación y demanda, meteorología, indisponibilidades y otras señales externas. Estas variables no son controladas por el agente, pero modifican el valor esperado de sus acciones.

**b) Estado interno de la posición o del activo.** En un problema de trading sin activo físico puede incluir posición acumulada, energía pendiente, exposición por mercado o límites de compra y venta. Cuando existe almacenamiento o generación gestionable, se añaden variables como estado de carga, potencia disponible, rampas, costes marginales y restricciones técnicas.

**c) Información de decisiones previas.** En problemas multi-etapa es esencial representar qué se ha comprado o vendido en mercados anteriores y qué posición residual permanece. Sin este bloque el agente puede tratar cada decisión como independiente, aunque económicamente forme parte de una misma trayectoria.

**d) Codificación temporal.** Hora, cuarto de hora, día de la semana, estacionalidad y tiempo restante hasta el cierre pueden codificarse de forma numérica o cíclica. La elección debe respetar estrictamente el principio *point-in-time*: ninguna variable puede contener información que no estuviera disponible en el instante en que se simula la decisión.

La inclusión de memoria explícita mediante redes recurrentes o extractores temporales puede ser útil cuando la observación instantánea no resume toda la información relevante. No obstante, una arquitectura más compleja solo está justificada si mejora de forma reproducible el desempeño fuera de muestra; no debe asumirse que una LSTM o un Transformer será superior por defecto.

### 3.3. Espacio de acciones

El espacio de acciones debe reflejar con la menor ambigüedad posible qué decisiones son realmente ejecutables.

En un **espacio discreto**, la política puede seleccionar entre acciones como comprar, vender o no operar, o elegir entre un conjunto finito de estrategias. Este diseño facilita el entrenamiento y la interpretación, pero puede perder precisión si la magnitud de la posición es económicamente relevante.

En un **espacio continuo**, la acción puede representar volumen, variación de posición, potencia de carga/descarga o parámetros de precio. DDPG, TD3 y SAC fueron desarrollados específicamente para control continuo <a href="#ref-8">[8]</a>–<a href="#ref-10">[10]</a>, mientras que PPO también se utiliza con frecuencia en espacios continuos mediante políticas estocásticas <a href="#ref-7">[7]</a>.

Una tercera alternativa es la **acción estructurada**, por ejemplo un vector de bloques precio–cantidad o una combinación de volumen y agresividad de ejecución. Este diseño es más próximo a la operativa de mercado, pero requiere un simulador capaz de traducir la acción en una probabilidad o resultado de casación. Cuando esa capa no existe, atribuir a la política una capacidad de ejecución que el entorno no modela produce una estimación optimista del rendimiento.

### 3.4. Función de recompensa

La recompensa es el vínculo entre el objetivo económico u operativo y el comportamiento aprendido. Una formulación genérica para trading eléctrico puede escribirse como:

**r_t = ΔPnL_t − λ_imb C_imb,t − λ_risk C_risk,t − λ_op C_op,t,**

donde ΔPnL representa el resultado económico incremental, *C_imb* el coste asociado al desvío o posición residual, *C_risk* un término de riesgo y *C_op* penalizaciones por acciones operativamente inviables.

La recompensa no debe mezclar términos sin una interpretación económica o operacional clara. Si se incorporan medidas de riesgo de cola, el Value at Risk (VaR) resume un cuantil de pérdidas, mientras que el Conditional Value at Risk (CVaR) cuantifica la pérdida media en la cola más desfavorable. El CVaR dispone de una formulación de optimización bien establecida <a href="#ref-4">[4]</a> y ha sido utilizado en trabajos de decisión secuencial sensible al riesgo <a href="#ref-5">[5]</a>.

Una estrategia útil consiste en definir la recompensa de forma **incremental respecto de un baseline**, de modo que el agente sea premiado por mejorar una política de referencia y no simplemente por capturar componentes de beneficio que existirían aunque no operase. Esta idea es especialmente relevante cuando el entorno contiene componentes de resultado económico independientes de la acción.

### 3.5. Restricciones: penalización frente a proyección

Las restricciones pueden tratarse de tres maneras. La primera es penalizar violaciones dentro de la recompensa. La segunda es **recortar o proyectar** la acción a la región factible antes de ejecutarla. La tercera es diseñar el espacio de acciones de modo que solo genere decisiones válidas. En aplicaciones industriales, las dos últimas opciones son preferibles para restricciones duras —límites de potencia, energía, posición o mercado—, mientras que las penalizaciones son apropiadas para preferencias blandas como reducir rotación, exposición o riesgo.

Debe distinguirse entre la acción propuesta por la política y la acción finalmente ejecutada por el entorno. Esta trazabilidad es necesaria para auditar por qué una política obtiene un resultado determinado y para evitar que el agente aprenda a explotar errores de modelado.

## 4. Algoritmos y arquitecturas relevantes

### 4.1. Métodos basados en valor, política y actor–critic

Los métodos **basados en valor** estiman funciones como *V(s)* o *Q(s,a)* y derivan la decisión a partir de esas estimaciones. DQN demostró la capacidad de combinar redes neuronales y Q-learning en espacios de acción discretos <a href="#ref-6">[6]</a>. Las arquitecturas *dueling* separan el valor del estado y la ventaja relativa de las acciones, lo que puede mejorar la estimación cuando varias acciones tienen efectos similares <a href="#ref-12">[12]</a>.

Los métodos **basados en política** optimizan directamente una distribución de acciones. PPO limita el tamaño efectivo de las actualizaciones mediante un objetivo sustituto con *clipping*, buscando un equilibrio entre estabilidad y simplicidad <a href="#ref-7">[7]</a>.

Los métodos **actor–critic** combinan una política (actor) y una estimación de valor (critic). DDPG extiende ideas de gradiente de política determinista a acciones continuas <a href="#ref-8">[8]</a>. TD3 introduce doble crítico, objetivos suavizados y actualizaciones retrasadas para reducir sesgos de sobreestimación <a href="#ref-9">[9]</a>. SAC adopta un objetivo de máxima entropía y aprendizaje *off-policy*, favoreciendo exploración y reutilización de experiencia <a href="#ref-10">[10]</a>. A2C es una variante síncrona de la familia actor–critic asociada al trabajo que popularizó A3C; no debe confundirse con “Asynchronous Advantage Actor-Critic”, denominación que corresponde a A3C <a href="#ref-11">[11]</a>.

### 4.2. Online RL y offline RL

En **online RL**, la política genera nueva experiencia al interactuar con un entorno real o simulado. PPO es *on-policy*: aprende principalmente con trayectorias producidas por la política actual o una versión muy próxima <a href="#ref-7">[7]</a>. SAC y TD3 son *off-policy* y pueden reutilizar transiciones almacenadas en un *replay buffer* <a href="#ref-9">[9]</a>, <a href="#ref-10">[10]</a>.

En **offline RL**, el aprendizaje se realiza a partir de un dataset estático sin interacción adicional durante el entrenamiento. La dificultad principal es el *distribution shift*: la política puede proponer acciones poco representadas en el dataset y el estimador de valor puede asignarles valores excesivamente optimistas. CQL aborda este problema mediante estimaciones conservadoras de Q <a href="#ref-13">[13]</a>, mientras que IQL evita evaluar explícitamente acciones fuera de la distribución mediante una formulación implícita <a href="#ref-14">[14]</a>. AWAC es especialmente relevante cuando se desea aprovechar datos previos y continuar posteriormente con ajuste online <a href="#ref-15">[15]</a>.

Es importante no confundir **offline RL** con entrenar un algoritmo online dentro de un simulador alimentado por datos históricos. En el segundo caso, el agente sí interactúa y genera transiciones en un entorno simulado; que las series exógenas procedan del histórico no convierte automáticamente el método en offline RL.

### 4.3. Arquitecturas neuronales y memoria temporal

Cuando el estado es un vector tabular de variables, una MLP constituye un baseline razonable por su simplicidad. Las redes recurrentes pueden aportar memoria cuando la observación es parcial o existe dependencia temporal no resumida por el estado. Los Transformers permiten modelar relaciones de largo alcance y se han incorporado a sistemas de trading y almacenamiento, pero su uso debe justificarse mediante comparación experimental frente a arquitecturas más sencillas.

Las CNN son adecuadas cuando la entrada posee estructura espacial o matricial; por ejemplo, mapas o representaciones densas del libro de órdenes. Los autoencoders pueden emplearse como reducción de dimensionalidad, pero añaden otra etapa de aprendizaje cuyo beneficio debe demostrarse.

### 4.4. Aprendizaje multiagente y jerárquico

En un entorno con participantes estratégicos, el problema puede formularse como aprendizaje multiagente. MADDPG introdujo un esquema de entrenamiento centralizado y ejecución descentralizada para entornos cooperativos y competitivos <a href="#ref-16">[16]</a>. MAPPO mostró que variantes de PPO pueden constituir baselines muy competitivos en tareas cooperativas <a href="#ref-17">[17]</a>.

El **modelado basado en agentes (ABM)** y el **aprendizaje multiagente (MARL)** están relacionados, pero no son sinónimos. ABM es un paradigma de simulación de sistemas formados por entidades autónomas con reglas de comportamiento; MARL se refiere específicamente a agentes que aprenden políticas mediante RL. Un ABM puede contener agentes no aprendices, y un sistema MARL no necesita adoptar toda la metodología típica de ABM.

El aprendizaje jerárquico introduce decisiones en diferentes escalas temporales —por ejemplo, una política de alto nivel que selecciona un modo operativo y otra de bajo nivel que decide volúmenes—. Aunque resulta atractivo para problemas complejos, incrementa la dificultad de entrenamiento y de atribución causal del rendimiento, por lo que conviene utilizarlo cuando exista una jerarquía operacional real.

### 4.5. Aprendizaje por imitación y regularización con el histórico

El histórico operativo puede utilizarse para algo más que comparar resultados. El *behavioral cloning* aprende una política supervisada a partir de pares estado–acción, proporcionando una inicialización plausible, aunque por sí solo reproduce el comportamiento observado y no optimiza explícitamente una recompensa.

DAgger reduce el error de distribución de la imitación supervisada agregando de manera iterativa estados visitados por la política y consultando la acción de un experto <a href="#ref-18">[18]</a>. GAIL aprende una política cuya distribución de trayectorias se aproxima a la de las demostraciones mediante un esquema adversarial <a href="#ref-19">[19]</a>, mientras que AIRL busca recuperar una función de recompensa más interpretable y transferible <a href="#ref-20">[20]</a>. En trading energético, estas técnicas son útiles principalmente como inicialización, restricción de soporte o benchmark; no sustituyen la evaluación económica fuera de muestra.

## 5. Implementación técnica y herramientas

### 5.1. Interfaces de entorno

La separación entre agente y entorno es crítica para la reproducibilidad. Gymnasium mantiene una API ampliamente utilizada para entornos de RL y formaliza métodos como `reset()` y `step()` <a href="#ref-21">[21]</a>. Para problemas multiagente, PettingZoo define interfaces específicas para interacción simultánea o secuencial entre múltiples agentes <a href="#ref-22">[22]</a>.

Esta separación permite cambiar el algoritmo sin reescribir la lógica de mercado, siempre que se mantengan estables las observaciones, acciones, transiciones y recompensas.

### 5.2. Frameworks de aprendizaje

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

### 5.3. Simulación de mercado, activos y red

Un entorno de trading energético puede separarse conceptualmente en cuatro capas: **mercado**, **posición/activo**, **liquidación** y **datos exógenos**. La capa de mercado determina qué acciones se consideran ejecutadas y a qué precio; la capa de activo impone restricciones físicas; la liquidación calcula PnL y desvíos; y los datos exógenos entregan la información disponible en cada instante.

ASSUME ofrece un marco abierto de simulación basada en agentes para mercados eléctricos e integra estrategias con aprendizaje por refuerzo <a href="#ref-29">[29]</a>. Para modelado físico, PyPSA permite análisis y optimización de sistemas energéticos <a href="#ref-30">[30]</a>, pandapower facilita flujo de potencia y otras funciones de análisis de redes eléctricas <a href="#ref-31">[31]</a>, y MATPOWER continúa siendo una referencia abierta para estudios de flujo de potencia y *optimal power flow* <a href="#ref-32">[32]</a>. Estas herramientas no son intercambiables: su utilidad depende de si el objetivo del proyecto exige simular clearing de mercado, restricciones del activo, red eléctrica o una combinación de esas capas.

Para un caso de trading mayorista sin impacto físico explícito en red, un gemelo digital mínimo puede consistir en: (i) reglas de secuenciación de mercados y ventanas de decisión; (ii) estado de posición; (iii) modelo de ejecución o casación; (iv) liquidación económica y desvíos; y (v) restricciones operativas. Incluir un simulador de red completo sin que las restricciones de red afecten realmente a las decisiones puede añadir complejidad sin mejorar la validez del experimento.

## 6. Particularidades de los mercados eléctricos y del contexto ibérico

Los mercados eléctricos presentan estacionalidad, picos de precio, restricciones físicas y una dependencia directa de la incertidumbre de demanda y renovables. Estas características han motivado aplicaciones de RL tanto en oferta como en despacho y gestión de carteras <a href="#ref-2">[2]</a>, <a href="#ref-3">[3]</a>.

En el ámbito ibérico existe información pública que puede alimentar un entorno de investigación. OMIE publica precios y resultados del mercado diario, curvas agregadas de oferta y demanda y resultados de mercados intradiarios, entre otros ficheros y visualizaciones <a href="#ref-33">[33]</a>. No obstante, la disponibilidad pública de curvas agregadas no equivale a disponer del libro de órdenes individual ni de la intención de cada participante; el modelo de ejecución debe respetar esa limitación.

La granularidad temporal también es relevante. La regulación española estableció un periodo de liquidación de desvíos de 15 minutos desde el 1 de diciembre de 2024 y adaptó posteriormente los procedimientos a la negociación cuarto-horaria en los mercados diario e intradiario <a href="#ref-34">[34]</a>. Por tanto, un sistema que pretenda representar la operación española actual debe alinear productos, posiciones y liquidaciones con esa granularidad cuando corresponda.

Desde el punto de vista de RL, estas reglas implican que el coste del desvío no es un elemento accesorio: forma parte de la función objetivo y puede modificar el valor de una operación intradiaria. La política debe aprender el compromiso entre capturar oportunidades de precio y reducir exposición residual.

## 7. Evidencia aplicada en trading y mercados eléctricos

### 7.1. Construcción de curvas de oferta en mercado diario

Di Persio, Garbelli y Giordano proponen un enfoque con DDPG para generar curvas escalonadas de oferta en un mercado diario utilizando datos históricos del mercado eléctrico italiano <a href="#ref-35">[35]</a>. El interés metodológico del trabajo reside en que la acción no es una simple señal de compra o venta, sino una decisión continua y multidimensional que representa una curva de oferta. Los autores reportan hasta aproximadamente un 85 % de recompensa normalizada en sus experimentos, definida respecto a un máximo teórico bajo sus hipótesis <a href="#ref-35">[35]</a>.

El trabajo muestra que una acción precio–cantidad puede aprenderse con actor–critic continuo. Al mismo tiempo, pone de relieve que el resultado solo es interpretable si el entorno representa cómo la oferta se transforma en energía casada y beneficio; de lo contrario, la política termina optimizando una abstracción del mercado.

### 7.2. Almacenamiento multi-mercado con información temporal

Li *et al.* presentan TempDRL, un enfoque para participación de baterías en mercado spot y mercados de reserva de contingencia del National Electricity Market australiano <a href="#ref-36">[36]</a>. El sistema combina un extractor temporal basado en Transformer con un agente de RL y se valida con precios reales. El estudio reporta mejoras frente a baselines de optimización y DRL, y muestra que la participación conjunta puede generar más valor que tratar los mercados de forma aislada <a href="#ref-36">[36]</a>.

Para un entorno con almacenamiento, el punto clave es que el SoC y las decisiones ya tomadas formen parte del estado: comprometer capacidad en un servicio limita lo que puede hacerse después. Una representación temporal más rica puede ayudar cuando existen dependencias persistentes en precios, pero esa mejora debe comprobarse frente a modelos más sencillos.

### 7.3. Renovables y almacenamiento en múltiples mercados

Cardo-Miota *et al.* formulan la operación conjunta de un sistema renovable con batería como un MDP y aplican TD3 a la participación en mercado day-ahead y servicios de reserva del programa DS3 en Irlanda <a href="#ref-37">[37]</a>. El caso utiliza datos históricos de una planta fotovoltaica de 7 MW y una batería de 1 MWh. Los autores reportan beneficios económicos adicionales frente a estrategias de referencia en los escenarios evaluados <a href="#ref-37">[37]</a>.

El interés para otros casos está en integrar restricciones físicas y participación en varios mercados dentro del mismo problema de decisión. Separar los ingresos por mercado y comparar con baselines permite comprobar si la mejora procede de la política y no de supuestos favorables del entorno.

### 7.4. Trading intradiario continuo para generación fotovoltaica

Abate *et al.* plantean un agente PPO para trading intradiario continuo de generación fotovoltaica con una recompensa que combina resultado de trading y exposición a desbalances <a href="#ref-38">[38]</a>. El trabajo utiliza información de mercado y previsiones dentro de un marco secuencial, incorpora un modelo de ejecución simplificado y evalúa la política fuera de muestra. La estrategia propuesta supera a los baselines considerados en los escenarios reportados, y el diseño enfatiza la interpretabilidad de la política <a href="#ref-38">[38]</a>.

Este planteamiento se acerca especialmente a un problema de ajuste de posición: la decisión no consiste solo en “comprar” o “vender”, sino también en determinar volumen y condiciones de ejecución mientras cambia la previsión de producción. Para el diseño del entorno destacan tres requisitos: reward relativo a un baseline, modelo de ejecución explícito y separación temporal estricta entre entrenamiento y evaluación.

### 7.5. Mercados locales, ABM y aprendizaje multiagente

Bâra y Oprea combinan modelado basado en agentes y RL para estudiar trading en mercados eléctricos locales con prosumidores y cargas flexibles <a href="#ref-39">[39]</a>. Comparan una estrategia que decide principalmente precio de oferta con otra que incorpora precio, cantidad y flexibilidad operativa. El trabajo reporta incrementos de ahorro financiero y de volumen negociado para la estrategia más rica en los escenarios evaluados <a href="#ref-39">[39]</a>.

Su transferencia directa a un mercado mayorista como OMIE es limitada, porque el mecanismo, la escala y los participantes son distintos. Sin embargo, el trabajo es una referencia útil para estudiar cómo la interacción entre múltiples agentes y el clearing de mercado puede cambiar el entorno que observa cada política.

### 7.6. Síntesis comparativa

**Tabla 2. Casos aplicados y elementos transferibles**

| Estudio | Mercado / activo | Algoritmo o enfoque | Acción | Elemento transferible |
|---|---|---|---|---|
| Di Persio *et al.* <a href="#ref-35">[35]</a> | Day-ahead, generador | DDPG | Curva precio–cantidad | Acción continua estructurada y modelado de casación. |
| Li *et al.* <a href="#ref-36">[36]</a> | Spot + FCAS, BESS | DRL + extractor temporal | Bidding multi-mercado | Estado temporal + SoC + coste de oportunidad entre mercados. |
| Cardo-Miota *et al.* <a href="#ref-37">[37]</a> | Day-ahead + reservas, PV+BESS | TD3 | Operación continua | Restricciones físicas integradas en un problema multi-mercado. |
| Abate *et al.* <a href="#ref-38">[38]</a> | Intradiario continuo, PV | PPO | Volumen / ejecución | Reward relativo a baseline, desvíos y validación temporal. |
| Bâra y Oprea <a href="#ref-39">[39]</a> | Mercado local, prosumidores | ABM + PPO | Precio, cantidad y flexibilidad | Interacción estratégica y clearing multiagente. |

No aparece un algoritmo dominante entre los trabajos revisados. Sí aparece un patrón común: acciones coherentes con la operativa, estado *point-in-time*, restricciones explícitas, simulación suficientemente realista y evaluación contra referencias.

## 8. Discusión: criterios de transferibilidad a un sistema de trading eléctrico

### 8.1. El entorno es parte del modelo

En RL, el entorno no es un mero generador de datos. Define qué consecuencias puede aprender la política. Si el entorno liquida todas las operaciones al precio histórico sin modelar ejecución, la política es esencialmente *price-taker*. Esta hipótesis puede ser válida para una primera prueba de concepto si se documenta, pero no permite concluir que una estrategia precio–cantidad funcionaría igual en producción.

Conviene, por tanto, trabajar por niveles de fidelidad. Un primer entorno puede validar la lógica secuencial de ajuste de posición y recompensa. Un segundo nivel puede incorporar precio límite o agresividad y una regla de casación basada en información histórica. Un tercer nivel, si el nivel de fidelidad requerido lo justifica y existen datos públicos adecuados, puede modelar libro de órdenes, impacto de mercado o interacción con otros agentes.

### 8.2. Dataset point-in-time y prevención de fuga de información

La validez de un backtest depende de reconstruir qué información estaba disponible en cada decisión. Las previsiones deben conservar su fecha de emisión; los precios de mercados posteriores no pueden aparecer en observaciones anteriores; y las transformaciones estadísticas deben ajustarse únicamente con el periodo de entrenamiento. Este principio afecta también a selección de hiperparámetros: el conjunto de test final debe permanecer aislado hasta que la política y las decisiones de diseño estén cerradas.

Un esquema recomendable es entrenamiento → validación temporal para selección → test final fuera de muestra. Para entornos no estacionarios puede añadirse evaluación *walk-forward*. La repetición con varias semillas aleatorias permite distinguir mejoras robustas de resultados dependientes de inicialización.

### 8.3. Baselines y atribución de valor

Un agente no debe evaluarse únicamente por PnL absoluto. Debe compararse, como mínimo, con: (i) política histórica; (ii) no operar o mantener posición; (iii) una regla heurística simple; y, cuando sea viable, (iv) una referencia de optimización con información comparable. La mejora económica debe reportarse junto con energía operada, coste de desvío, exposición, distribución de resultados y estabilidad temporal.

Cuando el agente cambia radicalmente el volumen negociado, una métrica de resultado total puede ser engañosa. Conviene añadir métricas normalizadas por energía y medidas de riesgo, además de analizar el peor desempeño por día, semana o periodo relevante.

### 8.4. Selección de algoritmo

Para una primera implementación con acciones continuas y un simulador estable, PPO, TD3 y SAC son candidatos razonables, pero responden a regímenes de datos distintos. PPO es simple y robusto en muchos entornos, aunque consume experiencia *on-policy*. TD3 y SAC reutilizan experiencia y suelen ser más eficientes en muestras. Si el proyecto dispone únicamente de un dataset fijo de transiciones y no de un simulador interactivo, CQL o IQL son opciones conceptualmente más adecuadas para offline RL <a href="#ref-13">[13]</a>, <a href="#ref-14">[14]</a>.

La selección debe realizarse mediante un protocolo común: mismo periodo de datos, mismas observaciones, mismas restricciones, misma función de recompensa, presupuesto de entrenamiento comparable y evaluación fuera de muestra. De lo contrario, atribuir la diferencia al algoritmo carece de validez experimental.

### 8.5. Reproducibilidad y gobierno del modelo

Un sistema destinado a investigación reproducible o soporte a la decisión debe registrar versión de datos, código, configuración del entorno, semilla, hiperparámetros, checkpoint y métricas. También debe permitir reconstruir una inferencia concreta: observación recibida, acción bruta, acción proyectada, motivo de cualquier restricción aplicada y resultado económico posterior.

La monitorización en producción debería diferenciar *drift* de datos, degradación de desempeño y cambio regulatorio. En mercados eléctricos, una modificación de reglas puede alterar el significado de una variable o la forma de liquidación, por lo que la actualización del entorno es tan importante como el reentrenamiento del agente.

## 9. Conclusiones

El aprendizaje por refuerzo proporciona un marco coherente para trading eléctrico porque representa explícitamente la secuencia de decisiones, el estado acumulado de la posición y las consecuencias económicas futuras. La literatura revisada muestra aplicaciones convincentes en ofertas day-ahead, almacenamiento multi-mercado, renovables con baterías, intradiario continuo y mercados locales. Sin embargo, esos resultados no justifican por sí solos la superioridad de RL frente a métodos tradicionales en cualquier contexto.

Los trabajos revisados apuntan a que el resultado depende tanto del entorno como del algoritmo. Elegir PPO, SAC o TD3 no compensa una representación deficiente de la información, las restricciones, la ejecución o la liquidación. Con fuga de información, reglas de casación irreales o una recompensa mal especificada, un backtest puede ser excelente y aun así carecer de valor operativo.

Para el contexto ibérico, la disponibilidad de precios y curvas agregadas de OMIE, junto con la liquidación cuarto-horaria de desvíos, ofrece una base para construir entornos de investigación realistas <a href="#ref-33">[33]</a>, <a href="#ref-34">[34]</a>. La línea de trabajo más sólida consiste en avanzar por niveles de fidelidad: comenzar con un agente *price-taker* que ajuste posiciones de forma secuencial, validar reward y backtesting, incorporar después una capa de ejecución precio–cantidad y, solo si los datos lo permiten, ampliar hacia simulación estratégica o multiagente.

Por último, cualquier afirmación de mejora debería sustentarse en datos fuera de muestra, baselines explícitos, métricas económicas y de riesgo, pruebas de sensibilidad y trazabilidad completa del experimento. Bajo estas condiciones, RL puede convertirse en una herramienta útil para apoyo a la decisión en mercados eléctricos; sin ellas, el principal riesgo es optimizar el simulador en lugar de mejorar la operación real.

## Referencias

<a id="ref-1"></a>[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

<a id="ref-2"></a>[2] Z. Zhu, Z. Hu, K. W. Chan, S. Bu, B. Zhou, and S. Xia, “Reinforcement learning in deregulated energy market: A comprehensive review,” *Applied Energy*, vol. 329, art. 120212, 2023, doi: 10.1016/j.apenergy.2022.120212.

<a id="ref-3"></a>[3] S. Giannelos, “Reinforcement Learning in Energy Finance: A Comprehensive Review,” *Energies*, vol. 18, no. 11, art. 2712, 2025, doi: 10.3390/en18112712.

<a id="ref-4"></a>[4] R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” *Journal of Risk*, vol. 2, no. 3, pp. 21–41, 2000, doi: 10.21314/JOR.2000.038.

<a id="ref-5"></a>[5] Y. Chow, A. Tamar, S. Mannor, and M. Pavone, “Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach,” in *Advances in Neural Information Processing Systems 28*, 2015, pp. 1522–1530.

<a id="ref-6"></a>[6] V. Mnih *et al.*, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, pp. 529–533, 2015, doi: 10.1038/nature14236.

<a id="ref-7"></a>[7] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017, doi: 10.48550/arXiv.1707.06347.

<a id="ref-8"></a>[8] T. P. Lillicrap *et al.*, “Continuous control with deep reinforcement learning,” arXiv:1509.02971, 2015, doi: 10.48550/arXiv.1509.02971.

<a id="ref-9"></a>[9] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 1587–1596.

<a id="ref-10"></a>[10] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 1861–1870.

<a id="ref-11"></a>[11] V. Mnih *et al.*, “Asynchronous Methods for Deep Reinforcement Learning,” in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 48, 2016, pp. 1928–1937.

<a id="ref-12"></a>[12] Z. Wang *et al.*, “Dueling Network Architectures for Deep Reinforcement Learning,” in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 48, 2016, pp. 1995–2003.

<a id="ref-13"></a>[13] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Q-Learning for Offline Reinforcement Learning,” in *Advances in Neural Information Processing Systems 33*, 2020.

<a id="ref-14"></a>[14] I. Kostrikov, A. Nair, and S. Levine, “Offline Reinforcement Learning with Implicit Q-Learning,” in *International Conference on Learning Representations (ICLR)*, 2022.

<a id="ref-15"></a>[15] A. Nair, A. Gupta, M. Dalal, and S. Levine, “AWAC: Accelerating Online Reinforcement Learning with Offline Datasets,” arXiv:2006.09359, 2020, doi: 10.48550/arXiv.2006.09359.

<a id="ref-16"></a>[16] R. Lowe *et al.*, “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” in *Advances in Neural Information Processing Systems 30*, 2017.

<a id="ref-17"></a>[17] C. Yu *et al.*, “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” in *Advances in Neural Information Processing Systems 35*, 2022.

<a id="ref-18"></a>[18] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,” in *Proc. 14th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 15, 2011, pp. 627–635.

<a id="ref-19"></a>[19] J. Ho and S. Ermon, “Generative Adversarial Imitation Learning,” in *Advances in Neural Information Processing Systems 29*, 2016.

<a id="ref-20"></a>[20] J. Fu, K. Luo, and S. Levine, “Learning Robust Rewards with Adversarial Inverse Reinforcement Learning,” in *International Conference on Learning Representations (ICLR)*, 2018.

<a id="ref-21"></a>[21] Farama Foundation, “Gymnasium Documentation.” https://gymnasium.farama.org/ (accessed Oct. 7, 2026).

<a id="ref-22"></a>[22] J. K. Terry *et al.*, “PettingZoo: Gym for Multi-Agent Reinforcement Learning,” in *Advances in Neural Information Processing Systems 34, Datasets and Benchmarks Track*, 2021.

<a id="ref-23"></a>[23] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

<a id="ref-24"></a>[24] E. Liang *et al.*, “RLlib: Abstractions for Distributed Reinforcement Learning,” in *Proc. 35th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 80, 2018, pp. 3053–3062.

<a id="ref-25"></a>[25] S. Huang, R. Dossa, C. Ye, and J. Braga, “CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms,” *Journal of Machine Learning Research*, vol. 23, no. 274, pp. 1–18, 2022.

<a id="ref-26"></a>[26] C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “MushroomRL: Simplifying Reinforcement Learning Research,” *Journal of Machine Learning Research*, vol. 22, no. 131, pp. 1–5, 2021.

<a id="ref-27"></a>[27] A. Bou *et al.*, “TorchRL: A data-driven decision-making library for PyTorch,” in *International Conference on Learning Representations (ICLR)*, 2024.

<a id="ref-28"></a>[28] J. Weng *et al.*, “Tianshou: A Highly Modularized Deep Reinforcement Learning Library,” *Journal of Machine Learning Research*, vol. 23, no. 267, pp. 1–6, 2022.

<a id="ref-29"></a>[29] N. Harder *et al.*, “ASSUME: An agent-based simulation framework for exploring electricity market dynamics with reinforcement learning,” *SoftwareX*, vol. 30, art. 102176, 2025, doi: 10.1016/j.softx.2025.102176.

<a id="ref-30"></a>[30] T. Brown, J. Hörsch, and D. Schlachtberger, “PyPSA: Python for Power System Analysis,” *Journal of Open Research Software*, vol. 6, no. 1, art. 4, 2018, doi: 10.5334/jors.188.

<a id="ref-31"></a>[31] L. Thurner *et al.*, “pandapower—An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems,” *IEEE Transactions on Power Systems*, vol. 33, no. 6, pp. 6510–6521, 2018, doi: 10.1109/TPWRS.2018.2829021.

<a id="ref-32"></a>[32] R. D. Zimmerman, C. E. Murillo-Sánchez, and R. J. Thomas, “MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education,” *IEEE Transactions on Power Systems*, vol. 26, no. 1, pp. 12–19, 2011, doi: 10.1109/TPWRS.2010.2051168.

<a id="ref-33"></a>[33] OMIE, “Resultados del Mercado: mercado diario e intradiario.” https://www.omie.es/ (accessed Oct. 7, 2026).

<a id="ref-34"></a>[34] Comisión Nacional de los Mercados y la Competencia, “Resolución de 6 de marzo de 2025 por la que se modifican los procedimientos de operación eléctricos para su adaptación a la negociación cuarto-horaria en los mercados diario e intradiario,” *Boletín Oficial del Estado*, no. 65, BOE-A-2025-5342, Mar. 17, 2025.

<a id="ref-35"></a>[35] L. Di Persio, M. Garbelli, and L. M. Giordano, “Reinforcement learning for bidding strategy optimization in day-ahead energy market,” *Energy Economics*, vol. 149, art. 108673, 2025, doi: 10.1016/j.eneco.2025.108673.

<a id="ref-36"></a>[36] J. Li, C. Wang, Y. Zhang, and H. Wang, “Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets,” *IEEE Transactions on Energy Markets, Policy and Regulation*, vol. 2, no. 3, pp. 392–406, 2024, doi: 10.1109/TEMPR.2024.3372656.

<a id="ref-37"></a>[37] J. Cardo-Miota, H. Beltran, E. Pérez, S. Khadem, and M. Bahloul, “Deep reinforcement learning-based strategy for maximizing returns from renewable energy and energy storage systems in multi-electricity markets,” *Applied Energy*, vol. 388, art. 125561, 2025, doi: 10.1016/j.apenergy.2025.125561.

<a id="ref-38"></a>[38] A. G. Abate, X. Liu, R. Liu, and X.-B. Zhang, “Feature-driven reinforcement learning for photovoltaic in continuous intraday trading,” arXiv:2510.16021, 2025, doi: 10.48550/arXiv.2510.16021.

<a id="ref-39"></a>[39] A. Bâra and S.-V. Oprea, “Trading Strategies on Local Electricity Markets Using Agent-Based Modelling and Reinforcement Learning: Vectors to Expand Energy Communities,” *Systems Research and Behavioral Science*, vol. 43, no. 3, pp. 1188–1211, 2026, doi: 10.1002/sres.70017.