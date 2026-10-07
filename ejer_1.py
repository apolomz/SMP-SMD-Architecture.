import numpy as np
import time
import concurrent.futures


ROWS, COLS = 1000, 1000
BLOCK_SIZE = 100

def compute_block_sum(matrix, r_start, r_end, c_start, c_end):
    
    ## Función que ejecuta cada hilo para sumar una submatriz
    return np.sum(matrix[r_start:r_end, c_start:c_end])

def main():
    # Semilla para reproducibilidad
    np.random.seed(42)
    
    # Creacion de la matriz con numeros aleatorios
    print(f"Generando matriz de {ROWS}x{COLS}...")
    matrix = np.random.rand(ROWS, COLS)

    # Programa secuencial
    start_seq = time.perf_counter()
    sum_seq = np.sum(matrix)
    end_seq = time.perf_counter()
    time_seq = (end_seq - start_seq) * 1000  

    print(f"\nEjecución-Secuencial")
    print(f"Suma Total: {sum_seq:.6f}")
    print(f"Tiempo:     {time_seq:.4f} ms")

    # Simulación SMP con hilos
    # Generar las coordenadas de los bloques de 100x100
    blocks = []
    for r in range(0, ROWS, BLOCK_SIZE):
        for c in range(0, COLS, BLOCK_SIZE):
            blocks.append((r, r + BLOCK_SIZE, c, c + BLOCK_SIZE))

    start_par = time.perf_counter()
    sum_par = 0.0

    # Usamos ThreadPoolExecutor para gestionar el pool de hilos SMP
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Enviamos cada bloque como una tarea a la cola del pool
        futures = [
            executor.submit(compute_block_sum, matrix, r_start, r_end, c_start, c_end)
            for r_start, r_end, c_start, c_end in blocks
        ]
        
        # Recolectamos y combinamos los resultados a medida que se completan
        for future in concurrent.futures.as_completed(futures):
            sum_par += future.result()

    end_par = time.perf_counter()
    time_par = (end_par - start_par) * 1000  

    print(f"\n--- Ejecución Paralela (SMP) ---")
    print(f"Suma Total: {sum_par:.6f}")
    print(f"Tiempo:     {time_par:.4f} ms")
    
    speedup = time_seq / time_par if time_par > 0 else 0
    print(f"\nSpeedup obtenido: {speedup:.2f}x")

if __name__ == "__main__":
    main()