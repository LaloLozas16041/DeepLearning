## Motivación y temas

Dentro de esta clase trabajaremos en conjuntos de datos del mundo real, para resolver problemas de negocios del día a día (Definitivamente no los aburridos conjuntos de datos de clasificación de iris o dígitos que vemos en todos los cursos). Aquí, trabajaremos en seis desafíos del mundo real:

- Redes Neurales Artificiales para resolver un problema de pérdida de clientes (Customer Churn)
- Redes neuronales convolucionales para el reconocimiento de imágenes
- Redes neuronales recurrentes para predecir los precios de las acciones
- Mapas auto-organizados para investigar el fraude
- Máquinas de Boltzmann para crear un Sistema de Recomendaciones
- Los AutoEncoders apilados* para enfrentarnos al reto por el premio del millón de dólares de Netflix

*Los AutoEncoders apilados son una nueva técnica del Deep Learning que hace un par de años aun no existía. Hasta ahora no hemos visto este método 

## Herramientas

TensorFlow y PyTorch son las dos librerías de código abierto más populares para el Deep Learning y en este curso aprenderás las dos.

TensorFlow fue desarrollado por Google y se utiliza en su sistema de reconocimiento de voz, en el nuevo producto de Google Photos, gmail, google search y mucho más. Las empresas que utilizan TensorFlow incluyen AirBnb, Airbus, Ebay, Intel, Uber y mcuhas más.

PyTorch es igual de potente y está siendo desarrollado por investigadores de Nvidia y universidades líderes: Stanford, Oxford, ParisTech. Las empresas que utilizan PyTorch incluyen Twitter, Saleforce y Facebook.

## Estudio de casos del mundo real

Dominar el Deep Learning no sólo consiste en conocer la intuición y las herramientas, sino también en ser capaz de aplicar estos modelos a escenarios del mundo real y obtener resultados reales y medibles para el negocio o proyecto. Es por ello que en este curso presentamos seis emocionantes retos:

### 1 Problema de modelización de la pérdida de clientes (Customer Churn)

En esta parte resolverás un desafío de análisis de datos para un banco. Te entregaremos un conjunto de datos con una gran muestra de los clientes del banco. Para crear este conjunto de datos, el banco recolectó información como la identificación del cliente, el puntaje de crédito, el género, la edad, la antigüedad, el balance, si el cliente está activo, tiene una tarjeta de crédito, etc. Durante un período de 6 meses, el banco observó si estos clientes abandonaban o permanecían en el banco.

El objetivo es hacer una Red Neural Artificial que pueda predecir, basándose en la información geo-demográfica y transaccional dada anteriormente, si algún cliente individual dejará el banco o se quedará (pérdida de clientes/Customer Churn)

Además, debes clasificar a todos los clientes del banco, basándose en su probabilidad de salida. Para ello, deberás utilizar el modelo correcto de Deep Learning, que se basa en un enfoque probabilístico.

Si tienes éxito en este proyecto, crearás un valor agregado significativo para el banco. Aplicando el modelo de Deep Learning, el banco puede reducir significativamente la tasa de pérdida de clientes (Customer Churn).

### 2 Reconocimiento de imágenes

En esta parte, crearás una Red Neuronal Convolucional que es capaz de detectar varios objetos en imágenes. Implementaremos este modelo del Deep Learning para reconocer un gato o un perro en un conjunto de imágenes. Sin embargo, este modelo también puede ser utilizado para detectar cualquier otra cosa y te mostraremos cómo hacerlo - simplemente cambiando las imágenes en la carpeta de entrada.

Por ejemplo, podrás entrenar el mismo modelo en un conjunto de imágenes del cerebro, para detectar si tienen un tumor o no. Pero si quieres mantenerlo adaptado a los gatos y perros, entonces podrás literalmente tomar una foto de tu gato o tu perro, y el modelo podrá predecir qué mascota tienes. Incluso lo hemos probado en el perro de Hadelin!

### 3 Predicción del precio de las acciones

En esta parte, crearás uno de los modelos más poderosos del Deep Learning. Incluso llegaremos a decir que crearás el modelo de Deep Learning más cercano a la "Inteligencia Artificial". ¿Por qué? Porque este modelo, como nosotros, tendrá memoria a largo plazo.

La rama del Deep Learning que facilita esto es la de las Redes Neuronales Recurrentes. Las RNN clásicas tienen poca memoria, y no eran ni populares ni poderosas por esta razón exacta. Pero una reciente e importante mejora en las Redes Neuronales Recurrentes dio lugar a la popularidad de las LSTM (Long Short Term Memory RNNs) que ha cambiado completamente el campo de juego. Estamos muy emocionados de incluir estos métodos de vanguardia del Deep Learning en nuestro curso.

En esta sección aprenderás a implementar este modelo extremadamente poderoso, y nosotros aceptaremos el reto de usarlo para predecir el precio real de las acciones de Google. Un desafío similar ya fue asumido por los investigadores de la Universidad de Stanford y nosotros trataremos de hacerlo al menos tan bien como ellos.

### 4 Detección de Fraude

Según un reciente informe publicado por Markets & Markets, el mercado de la detección y prevención de fraudes tendrá un valor de 33.19 billones de dólares para el año 2021. Se trata de una industria enorme y la demanda de habilidades avanzadas de Deep Learning seguirá creciendo. Por eso hemos incluido este caso de estudio en el curso.

Esta es la primera parte del Volumen 2 - Modelos de Deep Learning no supervisados. El reto empresarial aquí es la detección del fraude en las aplicaciones de tarjetas de crédito. Vas a crear un modelo de Deep Learning para un banco y te proporcionaremos un conjunto de datos que contiene información sobre los clientes que solicitan una tarjeta de crédito avanzada.

Estos son los datos que los clientes proporcionaron al llenar el formulario de solicitud. Tu tarea es detectar el fraude potencial dentro de estas aplicaciones. Esto significa que al final del desafío, literalmente obtendrás una lista explícita de los clientes que potencialmente hicieron trampa en sus solicitudes.

### 5 y 6 Sistemas de recomendación

Desde las sugerencias de productos de Amazon hasta las recomendaciones de películas de Netflix - los buenos sistemas de recomendación son muy valiosos en el mundo de hoy. Y los especialistas que pueden crearlos son algunos de los científicos de datos mejor pagados del planeta.

Trabajaremos en un conjunto de datos que tenga exactamente las mismas características que el conjunto de datos de Netflix: un montón de películas y miles de usuarios, que han valorado las películas que han visto. Las calificaciones van del 1 al 5, exactamente como en el conjunto de datos de Netflix, lo que hace que el Sistema de Recomendaciones sea más complejo de construir que en el caso de las calificaciones fueran que son sencillamente "Me Gusta" o "No me gusta".

Tu sistema de recomendación final será capaz de predecir las clasificaciones de las películas que los clientes no vieron. Por consiguiente, al clasificar las predicciones de 5 a 1, tu modelo de Deep Learning podrá recomendar qué películas debería ver cada usuario. Crear un sistema de recomendación tan poderoso es un gran reto, así que nos daremos dos oportunidades. Eso significa que lo construiremos con dos modelos diferentes de Deep Learning.

Nuestro primer modelo será Deep Belief Networks, máquinas complejas de Boltzmann que se abordarán en la Parte 5. Luego, nuestro segundo modelo será con los poderosos AutoEncoders, que personalmente, son mis favoritos. Podrás apreciar el contraste entre su simplicidad, y lo que son capaces de hacer.

E incluso podrás aplicarlo a ti mismo o a tus amigos. La lista de películas será explícita, así que sólo tendrás que calificar las películas que ya has visto, introducir tus calificaciones en el conjunto de datos, ejecutar tu modelo y el sistema de recomendación te dirá con exactitud qué películas te encantarán si no tienes idea de qué ver en Netflix.