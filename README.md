#  Turboshop - Entrevista Técnica

##  Objetivos

- Unificar datos de múltiples fuentes en un formato consistente.
- Construir las tablas necesarias para manejar la información de autos.
- Implementar una función para verificar si un auto ingresado existe en la base de datos.
- Detectar coincidencias aproximadas para corregir errores ortográficos o datos incompletos.
- Normalizar y limpiar los datos para asegurar comparaciones confiables.


## Estructura del proyecto

- Limpieza y preparación de datos por cada hoja del Excel en notebooks separados (con el mismo nombre que la hoja original).
- Generación de tablas consolidadas en `tables.ipynb`.
- Entrenamiento de modelos de red neuronal en la carpeta `train`.
- Generación y procesamiento de datos en la carpeta `generate_data`.
- Verificación y validación de autos en `check_car.ipynb`.


##  Cómo probar el proyecto

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/tuusuario/verificador-autos.git
   cd verificador-autos

2. Crear tu ambiente virtual de python e instalar las dependencias.

3. Abre check_car.ipynb para explorar las tablas y probar la función de verificación de autos.



