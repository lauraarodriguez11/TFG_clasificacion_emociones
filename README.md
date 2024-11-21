# Trabajo Fin de Grado: Clasificación de Emociones mediante Aprendizaje Automático

Este repositorio contiene el **Trabajo Fin de Grado (TFG)** de Laura Rodríguez Ropero, titulado *"Clasificación de Emociones mediante Aprendizaje Automático"*, realizado bajo la dirección del profesor José Manuel Perea Ortega, de la Universidad de Extremadura, en el año 2024.

## Resumen

El trabajo se centra en la clasificación de emociones a partir de textos cortos, como los encontrados en redes sociales, utilizando técnicas de **aprendizaje automático (ML)** y **aprendizaje profundo (DL)**. En el contexto del **Procesamiento de Lenguaje Natural (PLN)**, se exploran diferentes modelos, incluyendo **Redes Neuronales Recurrentes (RNN)**, **LSTM**, y **Transformers**, para categorizar emociones como ira, asco, miedo, alegría, tristeza, y sorpresa. Se presentan métodos de preprocesamiento de datos, normalización del texto, y técnicas avanzadas de optimización de modelos.

## Metodología

El trabajo se desarrolla a través de diversas fases que incluyen:

### 1. **Preprocesamiento de los datos**:
   - Limpieza de los datos textuales, eliminación de ruido (stopwords, puntuación, etc.).
   - Tokenización y lematización de los textos.
   - Conversión de los textos a representaciones numéricas utilizando **TF-IDF** y **Bolsa de Palabras**.

### 2. **Entrenamiento de los Modelos**:
   - **Modelos de Machine Learning (ML)**: Implementación de clasificadores tradicionales como **SVM** y **Random Forest**.
   - **Modelos de Deep Learning (DL)**: Utilización de **Redes Neuronales Recurrentes (RNN)** y **LSTM** para el análisis de secuencias de texto.

### 3. **Evaluación**:
   - Utilización de métricas como **Precisión**, **Recall**, **F1-Score** y **Accuracy** para evaluar la calidad de los modelos.
   - Comparación de los resultados obtenidos con modelos previos.

### 4. **Optimización**:
   - Ajuste de hiperparámetros utilizando técnicas de **GridSearch** y **RandomSearch**.
   - Regularización para evitar **Overfitting** y mejora en el rendimiento de los modelos.

## Archivos

Este repositorio contiene los siguientes archivos y directorios:

### 1. **TFG-Laura-Rodriguez-Ropero_signed.pdf**

Este archivo PDF contiene el **Trabajo Fin de Grado** completo, incluyendo el marco teórico, la metodología utilizada, los experimentos realizados y los resultados obtenidos. El documento detalla tanto la implementación de modelos de clasificación de emociones, como su evaluación.

- **Secciones importantes**:
  - **Capítulo 1**: Introducción y motivación del trabajo.
  - **Capítulo 2**: Marco teórico sobre Inteligencia Artificial, Aprendizaje Automático y Redes Neuronales.
  - **Capítulo 3**: Clasificación de emociones mediante aprendizaje profundo.
  - **Capítulo 4**: Aplicación práctica utilizando Python y los resultados de los experimentos.
  - **Conclusiones**: Resumen de hallazgos y posibles líneas de mejora.

### 2. **Archivos de código**: 
Los siguientes archivos contienen el código fuente implementado en **Python** para entrenar y evaluar los modelos de clasificación de emociones.

- **ML_1.ipynb**: Este notebook sigue un enfoque de aprendizaje automático tradicional, abarcando desde la preparación de datos hasta el entrenamiento y evaluación de modelos. Inicialmente, se realiza una limpieza exhaustiva del texto para normalizar los tweets, eliminando elementos irrelevantes como menciones y enlaces, además de convertir emojis en texto. Luego, se crean representaciones vectoriales de los textos utilizando TfidfVectorizer. Los modelos entrenados incluyen Multinomial Naive Bayes (MultinomialNB), Bernoulli Naive Bayes (BernoulliNB) y varias configuraciones de máquinas de soporte vectorial (SVC) con núcleos lineal, polinómico, radial (RBF) y sigmoide. Además, se entrena un Random Forest con profundidades de 20 y 50 árboles. La evaluación de los modelos se realiza mediante métricas como precisión, recall, F1-score y accuracy, cuyos resultados se presentan tanto en formato tabular como gráfico. Finalmente, los modelos más prometedores se exportan en archivos Pickle, junto con la bolsa de palabras generada, para ser reutilizados posteriormente.
  
- **ML_2.ipynb**: Este notebook es una continuación del anterior, enfocándose en la evaluación de los modelos previamente entrenados. Este notebook comienza cargando los modelos guardados en Pickle, que incluyen model_1 (SVC con núcleo sigmoide), model_2 (SVC con núcleo lineal), model_3 (SVC con núcleo RBF) y model_4 (Random Forest con 500 estimadores). También carga un conjunto de datos de prueba, que pasa por un proceso de transformación similar al de los datos de entrenamiento. Los modelos se utilizan para predecir emociones en los textos de prueba y sus predicciones se comparan con las etiquetas reales mediante métricas de clasificación como el reporte de clasificación y F1-score. Este análisis permite evaluar el rendimiento de los modelos en datos no vistos y determinar cuál es el más adecuado para tareas futuras.

- **DL.ipynb**: Este notebook se centra en la clasificación de emociones utilizando un modelo de Deep Learning. Específicamente, emplea XLM-Roberta, un modelo multilingüe preentrenado basado en transformers, para realizar análisis de emociones en textos. El flujo de trabajo incluye la importación de librerías clave como transformers, torch y tqdm, además de la carga de conjuntos de datos etiquetados relacionados con el análisis emocional. También se define una función de preprocesamiento que normaliza los textos eliminando menciones y enlaces. Posteriormente, se utiliza el modelo mediante el pipeline de transformers para predecir emociones y las predicciones se evalúan usando métricas como un reporte de clasificación y la matriz de confusión, cuya visualización ayuda a analizar el rendimiento.

### 3. **Archivos de datos**: 
Los siguientes archivos contienen los datos utilizados para entrenar y evaluar los modelos de clasificación de emociones.

- **emoevales_train.tsv**: Conjunto de datos de entrenamiento, que incluye ejemplos de textos con etiquetas emocionales para entrenar los modelos.
  
- **emoevales_dev.tsv**: Conjunto de datos de desarrollo (validación), utilizado para ajustar y validar los modelos durante el proceso de entrenamiento.
  
- **emoevales_test.tsv**: Conjunto de datos de prueba para evaluar el rendimiento final de los modelos entrenados.

- **emoevales_test_gold.tsv**: Conjunto de datos de prueba con las etiquetas correctas (de referencia) para comparar las predicciones del modelo con la verdad conocida.