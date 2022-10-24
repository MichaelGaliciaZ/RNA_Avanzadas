**Instituto Politécnico Nacional**

**Centro de Investigación en Computación**

**Redes Neuronales Avanzadas**

Nombre completo: Miguel Aarón Galicia Zainos

Fecha de entrega: 24 de October de 2022

**Tarea 3: Clasificador de Personajes con CNN**

# Descripción del problema

Es necesario realizar una red neuronal convolucional en conjunto con un clasificador para identificar personajes de Springfield. Esto se realizará por medio del dataset [https://www.kaggle.com/jfgm2018/the-simpsons-dataset-compilation-49-characters](https://www.kaggle.com/jfgm2018/the-simpsons-dataset-compilation-49-characters)

Para esto, se deberá hacer un notebook en el cual se realicen las siguientes etapas

1. Importar el data set y preprocesarlo (esto puede hacerse en un notebook adicional o en el mismo donde hagan la implementación de la CNN). El preprocesamiento debe incluir la generación de los archivos de los lotes de entrenamiento, ya sea en pickle o en numpy.
2. Diseñar una arquitectura convolucional con un clasificado para identificar a los diferentes ciudadanos de Springfield

Puntos por tomar en cuenta:

- Deben de usar los dataset para entrenamiento y para prueba. En Kaggle, vienen divididos, pero pueden combinarlos y rehacerlos como ustedes consideren necesario.
- Deben de seleccionar un mínimo de 10 personajes, como máximo el dataset completo con 49 clases.
- Pueden cambiar el tamaño de las imágenes en preprocesamiento
- Es necesario que el programa se elabore en TensorFlow 2

# Descripción del Dataset

El dataset fue creado originalmente por Alexandre Attia en 2017 mediante capturas de imagen en video, posteriormente se agregaron más imágenes por otros usuarios de la plataforma Kaggle.

|
 | Nombre | Train Image | Test Image |
| --- | --- | --- | --- |
| 0 | homer\_simpson | 4128 | 306 |
| --- | --- | --- | --- |
| 1 | bart\_simpson | 2562 | 261 |
| 2 | lisa\_simpson | 2383 | 222 |
| 3 | marge\_simpson | 1810 | 307 |
| 4 | ned\_flanders | 1454 | 49 |
| 5 | moe\_szyslak | 1452 | 50 |
| 6 | maggie\_simpson | 1371 | 162 |
| 7 | krusty\_the\_clown | 1206 | 50 |
| 8 | principal\_skinner | 1195 | 50 |
| 9 | charles\_montgomery\_burns | 1193 | 48 |
| 10 | milhouse\_van\_houten | 1091 | 49 |
| 11 | chief\_wiggum | 986 | 50 |
| 12 | abraham\_grampa\_simpson | 912 | 48 |
| 13 | sideshow\_bob | 877 | 47 |
| 14 | rainier\_wolfcastle | 757 | 28 |
| 15 | apu\_nahasapeemapetilon | 613 | 50 |
| 16 | mayor\_quimby | 523 | 50 |
| 17 | kent\_brockman | 498 | 50 |
| 18 | comic\_book\_guy | 469 | 49 |
| 19 | edna\_krabappel | 457 | 50 |
| 20 | waylon\_smithers | 407 | 30 |
| 21 | nelson\_muntz | 361 | 50 |
| 22 | lenny\_leonard | 310 | 50 |
| 23 | lionel\_hutz | 293 | 34 |
| 24 | cletus\_spuckler | 267 | 41 |
| 25 | barney\_gumble | 253 | 33 |
| 26 | gary\_chalmers | 236 | 931 |
| 27 | martin\_prince | 201 | 58 |
| 28 | gil | 197 | 52 |
| 29 | jimbo\_jones | 127 | 244 |
| 30 | groundskeeper\_willie | 122 | 46 |
| 31 | lunchlady\_doris | 117 | 226 |
| 32 | troy\_mcclure | 115 | 5 |
| 33 | dolph\_starbeam | 107 | 214 |
| 34 | selma\_bouvier | 103 | 31 |
| 35 | carl\_carlson | 98 | 31 |
| 36 | ralph\_wiggum | 90 | 38 |
| 37 | patty\_bouvier | 72 | 29 |
| 38 | miss\_hoover | 69 | 11 |
| 39 | professor\_john\_frink | 65 | 33 |
| 40 | kearney\_zzyzwicz | 61 | 122 |
| 41 | snake\_jailbird | 55 | 32 |
| 42 | brandine\_spuckler | 46 | 92 |
| 43 | agnes\_skinner | 42 | 26 |
| 44 | sideshow\_mel | 40 | 23 |
| 45 | otto\_mann | 39 | 19 |
| 46 | duff\_man | 30 | 60 |
| 47 | fat\_tony | 27 | 66 |
| 48 | disco\_stu | 8 | 77 |
| Total | 29895 | 4680 |

Como se aprecia en la Tabla anterior se cuenta con un total de 29895 imágenes para entrenamiento, sin embargo, no hay una cantidad homogénea de datos entre las clases, por ejemplo, para la clase _"homer\_simpson"_ se tienen 4128 imágenes para entrenamiento y en la clase _"disco\_stu"_ solo se cuentan con 8 imágenes. Los dataset desbalanceados provocan un sesgo en los resultados del modelo, por lo tanto, se deben tomar acciones para evitarlo. En general, estas acciones tendrán como objetivo aumentar los datos. Los datos de test se encuentran mal distribuidos, por ejemplo, la clase _"troy\_mcclure"_ se tienen 115 elementos de entrenamiento, pero solo 5 elementos de prueba. Se puede apreciar en la Figura que las imágenes de cada clase tienen diferentes tamaños, relación de aspecto, el fondo puede ser de un color o tener una escena. En algunos casos, las imágenes pueden tener más de un personaje. Debido a lo anterior, se hará una redistribución de los datos de entrenamiento, de validación y de test, así como un preprocesamiento de los datos para realizar el modelo.

![](RackMultipart20221024-1-ti85jf_html_933d2b9fdf6fbbec.png)

Una exploración a fondo revela que el dataset cuenta con diferentes tipos de archivos y no solo imágenes, las extensiones encontradas son .JPG, .bmp, .gif, .jpeg, .jpg, .png.

Debido a lo anterior, se deben tomar otro tipo de acciones en el preprocesamiento de los datos; En principio, no importa la extensión de los archivos, siempre y cuando se pueda representar como matrices, también se encontró que hay una diferencia considerable en el tamaño de las imágenes, por ejemplo, la imagen más grande tiene dimensiones 1920x1080 mientras que las imágenes más pequeñas cuentan con dimensiones menores a 600x600.

Las dimensiones de cada imagen deben estandarizarse para homogeneizar los datos de entrada, hay que tener en cuenta que el preprocesamiento también debe realizarse en la etapa de inferencia del modelo.

# Solución propuesta

Se propone clasificar 10 clases del dataset:

|
 | Nombre | Train Image | Test Image | Total |
| --- | --- | --- | --- | --- |
| 1 | maggie\_simpson | 1371.0 | 162.0 | 1533.0 |
| --- | --- | --- | --- | --- |
| 2 | ned\_flanders | 1454.0 | 49.0 | 1503.0 |
| 3 | moe\_szyslak | 1452.0 | 50.0 | 1502.0 |
| 4 | krusty\_the\_clown | 1206.0 | 50.0 | 1256.0 |
| 5 | principal\_skinner | 1195.0 | 50.0 | 1245.0 |
| 6 | charles\_montgomery\_burns | 1193.0 | 48.0 | 1241.0 |
| 7 | gary\_chalmers | 236.0 | 931.0 | 1167.0 |
| 8 | milhouse\_van\_houten | 1091.0 | 49.0 | 1140.0 |
| 9 | chief\_wiggum | 986.0 | 50.0 | 1036.0 |
| 10 | abraham\_grampa\_simpson | 912.0 | 48.0 | 960.0 |

La distribución de los datos para entrenamiento y validación son reordenadas en conjuntos de entrenamiento, validación y test, como se muestra en la siguiente tabla:

|
 | Nombre | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| 0 | maggie\_simpson | 1101 | 276 | 154 |
| --- | --- | --- | --- | --- |
| 1 | ned\_flanders | 1081 | 271 | 151 |
| 2 | moe\_szyslak | 1080 | 271 | 151 |
| 3 | krusty\_the\_clown | 904 | 226 | 126 |
| 4 | principal\_skinner | 896 | 224 | 125 |
| 5 | charles\_montgomery\_burns | 892 | 224 | 125 |
| 6 | gary\_chalmers | 840 | 210 | 117 |
| 7 | milhouse\_van\_houten | 820 | 206 | 114 |
| 8 | chief\_wiggum | 745 | 187 | 104 |
| 9 | abraham\_grampa\_simpson | 691 | 173 | 96 |
|
 | Total | 9050 | 2268 | 1263 |

El conjunto de datos es preprocesado y empaquetado en archivos TFRecords para facilitar su manipulación, así como el entrenamiento de la red neuronal. Por cada set, se obtuvieron 15 archivos con extensión _.tfrec_. Cada archivo contiene la imagen codificada a uint8, la etiqueta numérica (0, 1, 2, etc.) y la etiqueta categórica correspondiente.

## Preprocesamiento de datos

Los datos se estandarizan a una resolución de 64x64 pixeles, manteniendo los 3 canales de la imagen. Si la imagen tiene dimensiones menores a las especificadas se realiza la operación de _reshape_ conservando la relación de aspecto de la imagen. Si la imagen es mayor en alguna dimensión (Largo o Ancho) se realiza la operación de _reshape_ a 84x84 y luego se aplica la operación de _crop_ o _padding_ según se requiera, este paso es importante porque permite preservar información. La Figura muestra el efecto que se tiene al realizar el preprocesamiento, debido a que se conserva la relación de aspecto en las imágenes, la Figura muestra bordes negros. Las imágenes guardadas en los archivos TFRecord no se encuentran normalizadas. No se realiza ningún tipo de operación para el filtrado de imágenes, manipulación de contraste o brillo.

![](RackMultipart20221024-1-ti85jf_html_aeeb6854614585f6.png)

## Hiper-parámetros y medidas de desempeño

El modelo es entrenado con los siguientes hiper-parámetros:

## Definición del modelo

La Figura muestra la arquitectura utilizada, se utilizan 4 capas convolucionales y una capa densa de 64 neuronas, así como la capa de salida. Cada capa convolucional pasa por la función MaxPool de tamaño 2x2. El tamaño del filtro para cada capa convolucional es de 3x3. El stride en cada caso es de 1. La función de activación ReLu es usada en todas las capas internas, la capa de salida no tiene activación puesto que en el calculo del gradiente se aplica la función SoftMax.

## Entrenamiento

El modelo fue entrenado 13 épocas con batches de 40 imágenes. Los pesos iniciales fueron obtenidos mediante la inicialización de Lecun Uniform. El entrenamiento duro 5 minutos con una GPU Tesla T4.

![](RackMultipart20221024-1-ti85jf_html_5f4d27b884c86bde.png)

Se obtuvieron las curvas de accuracy y loss, las cuales se presentan en la Figura y Figura respectivamente.

![](RackMultipart20221024-1-ti85jf_html_3c14203a1da9d8b2.png)

## Evaluación del modelo

El accuracy obtenido en test fue de 0.87, considerablemente alto.

![](RackMultipart20221024-1-ti85jf_html_699e699231a05661.png)

Se obtuvo la matriz de confusión para visualizar el desempeño del modelo, así como las métricas que proporciona la librería sklearn con el método clasfication\_report.

![](RackMultipart20221024-1-ti85jf_html_e4ae9e3acd435a28.png)

# Comentarios finales

- El modelo a pesar de tener una estructura compleja es fácilmente entrenado en GPU's
- Las graficas de la Figura muestran un claro sobreajuste del 10% de la escala total, se debe reconsiderar usar métodos de regularización o bien reducir la complejidad del modelo.
- El uso de archivos externos es útil cuando se requiere minimizar los recursos computacionales, en lugar de cargar el conjunto de datos en memoria, ahora solo se carga una fracción de él, este procedimiento en conjuntos de datos mas grandes puede llegar a ralentizar el proceso de entrenamiento, además de que se requiere realizar un paso extra en el flujo de trabajo y si no se realiza adecuadamente la distribución de los datos así como su preprocesamiento, se puede llegar a necesitar varias versiones del conjunto de datos.
- Los pesos iniciales del modelo han sido cruciales para obtener un buen desempeño, puesto que en experimentos anteriores con pesos iniciales obtenidos mediante una distribución normal uniforme se requerían hasta 300 épocas para alcanzar una fracción del desempeño actual, aun con la misma arquitectura. Agradezco al Lic. Fernando A. Canto por su recomendación para usar inicializadores como Xavier, He y Lecun.
- Como se aprecia en la Figura la clase charles\_montgomery\_burns es la clase con más falsos positivos, siendo la clase moe\_szyslak la clase con la cual se confunde más. Este tipo de confusión se debe a que ambas clases comparten características similares, visten con el mismo tono de colores además de que su cabello es del mismo color.

Página 14 de 14
