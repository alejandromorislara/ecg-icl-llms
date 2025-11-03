# Análisis de ECGs con Grandes Modelos de Lenguaje: Estudio y Aplicación de ICL

## 📋 Descripción del Proyecto

Este proyecto investiga la aplicación de **In-Context Learning (ICL)** para el análisis de electrocardiogramas (ECG) utilizando modelos de lenguaje multimodales de código abierto. El objetivo principal es evaluar hasta qué punto los modelos open-source pueden adaptarse a tareas de interpretación de ECG sin acceso previo a datos de esta distribución específica, manteniendo la privacidad de datos médicos sensibles mediante despliegue local.

### Motivación

Los modelos propietarios como GPT-4o demuestran un buen rendimiento en tareas de interpretación de ECG, probablemente debido a exposición durante el entrenamiento. Sin embargo, su uso requiere enviar datos médicos sensibles a servidores externos, lo cual plantea serias preocupaciones de privacidad que los hospitales no están dispuestos a aceptar.

**Este trabajo explora alternativas de código abierto que pueden ejecutarse localmente**, protegiendo la privacidad de los datos mientras se investigan técnicas para compensar la falta de exposición previa a ECGs.

## 🎯 Hipótesis de Investigación

### Premisa Principal
El rendimiento de ICL se degrada significativamente cuando la distribución de datos difiere sustancialmente de los datos de entrenamiento del modelo.

### Hipótesis a Validar
1. **ICL solo**: Los modelos open-source sin exposición previa a ECGs mostrarán bajo rendimiento en tareas de interpretación
2. **ICL + CBM**: Introducir razonamiento explícito mediante Concept Bottleneck Models puede mejorar interpretabilidad y rendimiento
3. **Fine-tuning conceptual**: Un ajuste fino enfocado únicamente en conceptos básicos de ECG (ondas P, Q, R, S, T y escala) sin etiquetas diagnósticas puede habilitar el aprendizaje efectivo mediante ICL
4. **Recuperación post-fine-tuning**: El modelo ajustado debería recuperar el rendimiento en distribuciones previamente problemáticas

## 🔬 Pipeline Experimental

El proyecto sigue un pipeline de 6 etapas progresivas:

### Fase Preliminar: Toy Experiment
Validación de la premisa usando una tarea simplificada (secuencias simbólicas) para verificar que:
- ICL falla cuando los datos no siguen patrones familiares
- Fine-tuning básico resuelve esta limitación

### Fase Principal: Experimentos con ECGs Reales

1. **Baseline: MedGemma + ICL**
   - Evaluación del modelo base MedGemma con In-Context Learning
   - Métricas de rendimiento en tareas de clasificación de ECG
   - Análisis de casos de fallo

2. **MedGemma + ICL + CBM**
   - Integración de Concept Bottleneck Models
   - Razonamiento explícito sobre conceptos interpretables
   - Comparación de interpretabilidad vs baseline

3. **Fine-tuning Conceptual**
   - Ajuste fino en conceptos básicos de ECG **sin etiquetas diagnósticas**
   - Enfoque: ondas (P, Q, R, S, T), intervalos, y escala del papel
   - Novedad: evitar ruido de etiquetas diagnósticas en el ajuste

4. **Modelo Fine-tuned + ICL**
   - Evaluación del modelo ajustado con ICL
   - Comparación con baseline (etapa 1)

5. **Modelo Fine-tuned + ICL + CBM**
   - Combinación completa de técnicas
   - Evaluación final de rendimiento e interpretabilidad

## 🗂️ Estructura del Proyecto


## 🚀 Guía de Uso

### 1. Instalación

```bash
# Crear entorno conda
conda env create -f environment.yml
conda activate ecg-icl

# O usar pip
pip install -r requirements.txt
```

### 2. Toy Experiment (Validación de Premisa)

#### Generar datos sintéticos
```bash
python scripts/generate_toy_dataset.py --n-test-samples 999 --n-ood-samples 300
```

Esto genera:
- 24 ejemplos ICL (8 por clase)
- 999 ejemplos de test in-distribution
- 300 ejemplos de test out-of-distribution

#### Evaluar ICL

**Nota**: Necesitas un servidor LLM local compatible con OpenAI API (ej: LM Studio, llama.cpp)

```bash
# Zero-shot
python scripts/evaluate.py --task 1 --n-shots 0

# Few-shot (4 ejemplos)
python scripts/evaluate.py --task 1 --n-shots 4

# Evaluar en datos OOD
python scripts/evaluate.py --task 1 --n-shots 4 --ood

# Con modelo específico
python scripts/evaluate.py --task 1 --n-shots 8 --model-name "medgemma-2b"
```

### 3. Preprocesar Datos Reales (PTB-XL)

```bash
# Descargar y preprocesar PTB-XL
python scripts/preprocess_ptbxl.py --data_dir data/raw/PTBXL --output_dir data/processed/ptbxl
```

### 4. Experimentos Principales

#### ICL con MedGemma
```bash
python scripts/evaluate.py --config configs/medgemma_icls.yaml
```

#### Entrenar CBM
```bash
python scripts/train_cbm.py --config configs/cbm_config.yaml
```

#### Fine-tuning Conceptual
```bash
python scripts/finetune_model.py --config configs/medgemma_finetune.yaml
```

## 📊 Datasets

### Toy Experiment
- **Tipo**: Secuencias simbólicas sintéticas
- **Alfabeto**: `{.|:_~}`
- **Tarea**: Clasificación de frecuencia cardíaca en 3 clases
- **Propósito**: Validar hipótesis de ICL en entorno controlado

### PTB-XL
- **Tipo**: ECGs reales de 12 derivaciones
- **Muestras**: ~21,800 registros
- **Fuente**: [PhysioNet](https://physionet.org/content/ptb-xl/)
- **Tareas**: Clasificación de diagnósticos cardíacos

*Ver `docs/datasets.md` para más detalles*

## 🔧 Configuración de Modelo Local

Para ejecutar los experimentos necesitas un servidor LLM local. Opciones recomendadas:

### LM Studio (Recomendado para principiantes)
1. Descargar [LM Studio](https://lmstudio.ai/)
2. Cargar un modelo (ej: Llama 3, Mistral, MedGemma)
3. Iniciar servidor local (por defecto: `http://127.0.0.1:1234/v1`)

### Experimentos Principales
*Sección en progreso - se actualizará con resultados*

## 📚 Documentación Adicional

- [Metodología detallada](docs/methodology.md)
- [Plan de experimentos](docs/experiments_plan.md)
- [Consideraciones de privacidad y ética](docs/privacy_ethics.md)
- [Referencias](docs/references.md)

## 🔐 Privacidad y Ética

Este proyecto prioriza la privacidad de datos médicos:
- Todos los modelos pueden ejecutarse **completamente en local**
- No se requiere conexión a APIs externas
- Compatible con entornos hospitalarios con restricciones de seguridad
- Datos sensibles nunca salen del servidor local

Ver `docs/privacy_ethics.md` para consideraciones detalladas.


## 📄 Licencia

*Por determinar*

## 📖 Citación

*Por determinar*

## 🤝 Contribuciones

Este es un proyecto de investigación académica (TFG). 

---

**Última actualización**: Noviembre 2025
