# Jhoan Sebastian Fernandez
# 2222772

Este repositorio contiene los ejercicios del taller 4, ejercicios en paralelo utilizando Python y NumPy para evaluar arquitecturas de procesamiento masivo.

## Ejercicios

* **Ejercicio 1 (SMP):** Simula un sistema multihilo en memoria compartida para sumar una matriz dividida en bloques, analizando el costo administrativo (*overhead*) de gestión de hilos.
* **Ejercicio 2 (SIMD):** Compara la multiplicación de matrices entre bucles secuenciales en Python ($\mathcal{O}(N^3)$) y la aceleración vectorial mediante instrucciones SIMD de NumPy.
* **Ejercicio Integrador (Híbrido SMP-SIMD):** Combina hilos concurrentes (SMP) con reducción vectorial por columnas (SIMD) para procesar eficientemente una matriz masiva de $10000 \times 10000$.

## Requisitos y Ejecución

```bash
pip install numpy
python ejer_1.py
python ejer_2.py
python smp_smd.py