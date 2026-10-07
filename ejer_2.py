import numpy as np
import time

N = 1000 

def multiply_traditional(A, B):
    #Multiplicacion de matrices
    result = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] += A[i][k] * B[k][j]
    return result

def main():
    np.random.seed(42)
    print(f"Generando matrices A y B de {N}x{N}...")
    
    # Creacion de dos matrices de 1000x1000
    A_np = np.random.rand(N, N)
    B_np = np.random.rand(N, N)

    # Convertir a listas
    A_list = A_np.tolist()
    B_list = B_np.tolist()

    # 2. Multiplicación optimizada con SIMD (Numpy)
    start_simd = time.perf_counter()
    C_simd = A_np @ B_np  # Equivalente a np.dot(A_np, B_np)
    end_simd = time.perf_counter()
    time_simd = (end_simd - start_simd) * 1000  # ms

    print(f"Multiplicación NumPy (SIMD) completada en: {time_simd:.4f} ms")

    # Multiplicacion con bucles
    print("Ejecutando multiplicación con bucles...")
    start_trad = time.perf_counter()
    C_trad = multiply_traditional(A_list, B_list)
    end_trad = time.perf_counter()
    time_trad = (end_trad - start_trad) * 1000  # ms

    print(f"Multiplicación con bucles completada en: {time_trad:.4f} ms")

    # Calculo y comparacion
    speedup = time_trad / time_simd
    print(f"\nAceleración SIMD vs Multriplicacion: {speedup:.2f}x de rapidez")

if __name__ == "__main__":
    main()