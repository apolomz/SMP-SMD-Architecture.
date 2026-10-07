import numpy as np
import time
import concurrent.futures

ROWS, COLS = 10000, 10000
BLOCK_SIZE = 1000

def process_block_simd(matrix, r_start, r_end, c_start, c_end):

    # Función que ejecuta cada hilo (SMP).
    #  Aplica operaciones vectorizadas SIMD con NumPy dentro del bloque asignado.
    
    # Operacion SIMD
    column_sums = np.sum(matrix[r_start:r_end, c_start:c_end], axis=0)
    # Reducción 
    return np.sum(column_sums)

def main():
    np.random.seed(42)
    print(f"Generando matriz de {ROWS}x{COLS} ({ROWS*COLS:,} elementos)...")
    matrix = np.random.rand(ROWS, COLS)

    # Ejecución Secuencial
    start_seq = time.perf_counter()
    sum_seq = np.sum(matrix)
    end_seq = time.perf_counter()
    time_seq = (end_seq - start_seq) * 1000  

    print(f"\nEjecución Secuencial")
    print(f"Suma Total: {sum_seq:.6f}")
    print(f"Tiempo:     {time_seq:.4f} ms ({time_seq/1000:.4f} s)")

    # Ejecución SMP - SIMD
    # Generar las coordenadas de los bloques de 1000x1000
    blocks = []
    for r in range(0, ROWS, BLOCK_SIZE):
        for c in range(0, COLS, BLOCK_SIZE):
            blocks.append((r, r + BLOCK_SIZE, c, c + BLOCK_SIZE))

    start_par = time.perf_counter()
    sum_par = 0.0

    # SMP: Administrar distribución de bloques mediante pool de hilos
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_block_simd, matrix, r_start, r_end, c_start, c_end)
            for r_start, r_end, c_start, c_end in blocks
        ]
        
        for future in concurrent.futures.as_completed(futures):
            sum_par += future.result()

    end_par = time.perf_counter()
    time_par = (end_par - start_par) * 1000  # ms

    print(f"\n Ejecución SMP + SIMD")
    print(f"Suma Total: {sum_par:.6f}")
    print(f"Tiempo:     {time_par:.4f} ms ({time_par/1000:.4f} s)")

    # Comparación 
    speedup = time_seq / time_par if time_par > 0 else 0
    print(f"\nSpeedup Obtenido: {speedup:.2f}x de rapidez")

if __name__ == "__main__":
    main()