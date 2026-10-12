# Protocolo del experimento revisado

Autores: Luis Gonzalez (23353) y Jonathan Zacarias (231104).

## Datos y evaluacion

Se revisan visualmente los 312 PNG suministrados. Se aceptan 20 personajes de Tiny Dungeon y 179 criaturas de Tiny Creatures. Se excluyen escenarios, objetos, equipo aislado y una mano sin cuerpo. Se comprueban duplicados exactos y reflejos exactos. No se fuerza una cantidad objetivo.

La caja de contenido no negro se centra sobre un lienzo negro de 16 por 16, sin estirar el personaje. El PNG procesado se escala a 64 por 64 con NEAREST. Las familias de color, tamano o pose identificadas y las siluetas exactamente iguales permanecen en un mismo grupo. Con semilla 1729, se reserva el 20% de grupos para validacion: 164 sprites en entrenamiento y 35 en validacion. Solo entrenamiento recibe reflejo horizontal, para un total de 328 ejemplos. No se permutan los canales RGB.

La validacion se usa para evaluacion y seleccion de modelo. No participa en el calculo de gradientes. No existe un tercer conjunto de prueba independiente, por lo que las conclusiones sobre generalizacion deben ser limitadas.

## Condiciones controladas

Se usa una DCGAN que aprende a resolucion nativa de 16 por 16 y exporta 64 por 64 mediante NEAREST dentro del generador. Todas las condiciones comparten arquitectura, BatchNorm, inicializacion, Adam, tasa 0.0002, betas 0.5 y 0.999, lote 32, datos, numero de epocas, orden de lotes y ruido.

La base usa BCEWithLogitsLoss no saturante. Hinge cambia exclusivamente la perdida respecto a la base. Hinge con normalizacion espectral cambia exclusivamente la normalizacion de los pesos convolucionales del discriminador respecto a Hinge. Conserva BatchNorm para aislar el cambio. La funcion que crea la normalizacion espectral conserva los pesos convolucionales iniciales. Los hashes previos a aplicarla deben coincidir.

Las hipotesis se registran en cada carpeta antes del entrenamiento. Se guarda el ruido fijo de 16 vectores con semilla 123. El ruido de evaluacion usa semilla 777. La recalibracion de BatchNorm usa un generador aleatorio separado, semilla 12345 y 20 lotes de 128 vectores. No altera la secuencia de ruido ni el orden de lotes de entrenamiento.

## Exploracion y extension

Primera etapa: tres corridas de 300 epocas con semilla 42, almacenadas en runs/defendible. Las muestras del mismo ruido fijo conservaron estructuras difusas. Este resultado motiva una segunda etapa de 1500 epocas por condicion, desde la misma inicializacion. Se conservan datos y todos los hiperparametros. Las corridas de esta etapa se almacenan en runs/defendible_1500. La extension se registra antes de comenzar. Las curvas y resultados finales deben proceder de una sola etapa identificada; no se mezclan con los historicos.

La extension de BCE e Hinge sigue mostrando bordes difusos y dominancia de D. Se registra una tercera etapa, antes de ejecutarla: runs/paleta_2000, 2000 epocas por condicion desde cero. La arquitectura comun termina en 23 canales, uno por cada color que aparece en entrenamiento. Cada pixel recibe el color con mayor puntuacion. Durante entrenamiento se usa un gradiente sustituto de softmax (straight-through) para atravesar esa seleccion discreta. No se usan colores ni gradientes de validacion. La salida discreta participa tanto en entrenamiento como en evaluacion, y forma parte del generador; no se retocan los PNG finales. Esta modificacion es una nueva configuracion base compartida por las tres condiciones. Las comparaciones de perdida y estabilizacion se hacen dentro de la misma etapa, no entre etapas con distinta arquitectura. El estimador straight-through introduce una aproximacion de gradiente que debe declararse como limitacion. El cambio de arquitectura es exploratorio y no constituye una ablacion controlada de la paleta por si solo.

## Criterio para elegir y entregar

Se compara la calidad visual de los mismos vectores, la distancia media entre muestras, vecinos de entrenamiento, distancia a validacion y trayectoria de las perdidas. Se prefiere una condicion que produzca siluetas mas reconocibles, variedad de formas y fondos uniformes, sin parecido excesivo con ejemplos particulares. No se minimiza una sola perdida ni se elige por la magnitud de perdidas entre funciones diferentes.

La similitud coseno de ResNet-18 y el RMS son evidencia auxiliar. Se informa su sesgo hacia fondo y color. RMS menor que 0.02 se usa como alerta heuristica, no como prueba de ausencia de memorizacion. Un fondo negro compartido puede reducir RMS o elevar IoU sin que exista similitud semantica. No se interpreta la diversidad alta como calidad alta.

Se generan 200 candidatos por condicion final con semillas 1000 a 1199. La galeria final contiene 10 candidatos revisados con sus vecinos. Se declara 10/200 para esa seleccion y se revela que tambien se revisaron muestras de las otras condiciones y checkpoints. Los PNG finales proceden directamente del generador y solo reciben conversion a 8 bits. No se usa generacion externa ni retoque.

Se guarda el checkpoint exacto, su hash y cada vector z. Una comprobacion en CPU debe regenerar los 10 PNG con error menor que 2/255. Se incluyen datos originales, manifiestos, configuraciones, historiales, pesos, figuras y comandos de reproduccion. Una corrida por condicion y un conjunto pequeno no permiten probar ausencia de sobreajuste ni significacion estadistica.
# Exploración adicional antes de ejecutarla: tasa de D

11 de octubre, 20:14, Guatemala. Las tres corridas de paleta muestran dominancia de D y repetición de arquetipos. Creemos que reducir solamente la tasa de D de 0.0002 a 0.00002 puede mantener una señal de aprendizaje útil para G. Lo consideraremos útil si reduce la dominancia prolongada y mejora la legibilidad y variedad del ruido fijo. Se probará primero Hinge en `runs/paleta_tasa_D`, conservando 2000 épocas, pesos iniciales, datos, arquitectura, paleta, BatchNorm, ruido y tasa de G. Si justifica su selección, se completarán BCE y Hinge+SN con la misma tasa de D, para conservar los controles de los dos experimentos obligatorios. Esta decisión se documenta antes de iniciar la corrida; no se considerará una mejora por el solo hecho de obtener pérdidas menores.

Exploración antes de ejecutarla, 20:16: bajar la tasa de D conservó repetición y redujo diversidad de Hinge de 0.300 a 0.229; no se selecciona esa corrida. Creemos que reducir solo la capacidad de D (canales base 64 a 16; tasa original 0.0002) puede limitar su separación inmediata de un conjunto tan pequeño. Lo consideraremos útil si las muestras fijas resultan más legibles y variadas, sin confundir pérdidas bajas con calidad. Se prueba primero Hinge con 2000 épocas y salida de paleta en runs/paleta_D16. Si se selecciona, se completarán los controles BCE y Hinge+SN con la misma arquitectura compacta de D. La tasa permanece en su valor original para aislar la capacidad frente a runs/paleta_2000.

Cierre de exploraciones: reducir la capacidad de D en Hinge obtuvo diversidad 0.185 frente a 0.300 del control a época 2000, sin mejora visual defendible; no se adopta. Se elige un checkpoint anterior del control BCE (1500) por inspección de muestras y vecinos, y se comparan las tres condiciones a esa misma época. Las tres corridas se entrenaron por 2000 épocas. La primera exportación desde GPU mostró un píxel nativo que cambió de color al regenerar en CPU, debido al argmax discreto cerca de un empate. Se conserva esa verificación fallida como evidencia y se establece CPU, lote de inferencia 1, como procedimiento canónico para la galería final; se recalculan candidatos y vecinos bajo ese procedimiento.
