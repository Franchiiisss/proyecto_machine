# MovieLens Persona Studio - MLP para Like/Dislike

## Descripción del Proyecto

Sistema de recomendación basado en MLP que predice si un usuario dará like/dislike a una película, utilizando personas cinéfilas construidas mediante clustering.

**Dataset**: MovieLens 100K / 1M
- 100,000 ratings (1-5) de 943 usuarios sobre 1682 películas
- Like/Dislike: umbral ≥ 4 estrellas

## Estructura del Proyecto

```
proyecto_machine/
│
├── 1_procesamiento_datos.ipynb    # Carga, limpieza y feature engineering
├── 2_visualizacion.ipynb          # Análisis exploratorio y visualizaciones
├── 3_clustering.ipynb             # Construcción de personas cinéfilas
├── 4_mlp.ipynb                    # Entrenamiento y evaluación del MLP
│
├── proyecto.ipynb                 # Notebook completo original
│
├── data_processed/                # Datos procesados
│   ├── data_final.csv            # Dataset completo con features
│   ├── user_features.csv         # Features de usuarios
│   ├── movie_features.csv        # Features de películas
│   ├── personas_cinefilas.csv    # Resumen de personas
│   ├── model_results.csv         # Resultados de modelos
│   └── feature_importance.csv    # Importancia de features
│
├── ml-100k/                       # Dataset MovieLens 100K
├── ml-1m/                         # Dataset MovieLens 1M
│
├── mlp_model.pkl                  # Modelo MLP entrenado
├── scaler.pkl                     # StandardScaler
├── le_gender.pkl                  # LabelEncoder para género
├── le_occupation.pkl              # LabelEncoder para ocupación
│
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo

```

## Orden de Ejecución

Para ejecutar el proyecto completo, sigue este orden:

1. **1_procesamiento_datos.ipynb**
   - Carga de datos (MovieLens 100K o 1M)
   - Creación de variable objetivo (like/dislike)
   - Feature engineering (preferencias, embeddings SVD)
   - Guarda: `data_processed/data_final.csv`, `user_features.csv`, `movie_features.csv`

2. **2_visualizacion.ipynb**
   - Análisis exploratorio de datos
   - Visualizaciones de distribuciones
   - Análisis de embeddings (t-SNE, UMAP)
   - Detección de clusters con DBSCAN

3. **3_clustering.ipynb**
   - Comparación de algoritmos (K-Means, GMM, Spectral)
   - Selección de hiperparámetros
   - Construcción de 5 personas cinéfilas
   - Caracterización de cada persona
   - Guarda: `personas_cinefilas.csv`

4. **4_mlp.ipynb**
   - Entrenamiento de 3 modelos MLP (Baseline, Class Weights, SMOTE)
   - Evaluación con ROC-AUC, F1, PR-AUC
   - Matriz de confusión y análisis de errores
   - Métricas de ranking (Hit@K, NDCG@K)
   - Validación cruzada
   - Guarda: `mlp_model.pkl`, `model_results.csv`

## Instalación

```bash
pip install -r requirements.txt
```

## Características Principales

### Features de Usuario
- Preferencias por género (18 géneros)
- Preferencias por década
- Estadísticas (media, std, #votos, diversidad)
- Embeddings latentes SVD (20 componentes)
- Metadatos (edad, género, ocupación)

### Features de Película
- Géneros (one-hot encoding)
- Década normalizada
- Embeddings latentes SVD (20 componentes)

### Personas Cinéfilas
- 5 clusters identificados con K-Means
- Cada persona tiene perfil único de géneros y preferencias
- Caracterización por edad, géneros favoritos, películas representativas

## Resultados

### Mejor Modelo: MLP con SMOTE
- **ROC-AUC**: ~0.85
- **F1-Score**: ~0.80
- **PR-AUC**: ~0.83
- **Hit@10**: ~75%
- **NDCG@10**: ~0.70

### Arquitectura MLP
- Capas ocultas: (128, 64, 32)
- Activación: ReLU
- Optimizador: Adam
- Early stopping habilitado

## Cambiar entre Datasets

Para cambiar entre MovieLens 100K y 1M, modifica la variable `DATASET` en `1_procesamiento_datos.ipynb`:

```python
DATASET = '100k'  # Opciones: '100k' o '1m'
```

## Notas

- Los archivos `u1.base`, `u1.test`, etc. en `ml-100k/` NO se usan (son splits predefinidos)
- Nosotros creamos nuestros propios splits con `train_test_split`
- SMOTE se aplica solo en el conjunto de entrenamiento
- El modelo maneja cold-start usando metadatos de usuario y película

## Requisitos Cumplidos

- MLP binario (usuario, película) → like  
- Target binario (rating ≥ 4)  
- Manejo de desbalance (SMOTE + Class Weights)  
- ROC-AUC, F1-Score, PR-AUC  
- Matriz de confusión  
- Análisis de casos límite (FP/FN)  
- Hit@K y NDCG@K para Top-N  
- Estrategias cold-start con metadatos  
- Validación cruzada  
- Evaluación por usuario  
- Importancia de features  
- Personas cinéfilas (clustering)  
- Embeddings latentes (SVD)  
