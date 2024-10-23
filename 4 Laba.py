'''Лабораторная работа №4
С клавиатуры вводится два числа K и N. Квадратная матрица А(N,N), состоящая из 4-х равных по размерам подматриц,
B,C,D,E заполняется случайным образом целыми числами в интервале [-10,10].
Для отладки использовать не случайное заполнение, а целенаправленное. Вид матрицы А:
E B
C D
Вариант 14.	Формируется матрица F следующим образом: скопировать в нее А и если в В количество чисел, меньших К в нечетных столбцах больше, чем сумма чисел в четных строках, то поменять местами С и Е симметрично, иначе В и Е поменять местами несимметрично. 
При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,то вычисляется выражение: A-1*AT – K * F, иначе вычисляется выражение (A-1 +G-FТ)*K, где G-нижняя треугольная матрица, полученная из А.
Выводятся по мере формирования А, F и все матричные операции последовательно.
'''

import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(N):
    A = np.zeros((N, N))  
    for i in range(N):
        for j in range(N):
            if i >= N // 2 and j < N // 2:
                A[i, j] = np.random.randint(-10, 10)  # Подматрица B
            elif i >= N // 2 and j >= N // 2:
                A[i, j] = np.random.randint(-10, 10)  # Подматрица C
            elif i < N // 2 and j >= N // 2:
                A[i, j] = np.random.randint(-10, 10)  # Подматрица D
            else:
                A[i, j] = np.random.randint(-10, 10)  # Подматрица E
    return A

def form_matrix_F(A, K):
    N = A.shape[0]
    F = A.copy()

    # Разделение матрицы A на подматрицы B, C, D, E
    B = A[:N//2, N//2:]
    C = A[N//2:, N//2:]
    D = A[N//2:, :N//2]
    E = A[:N//2, :N//2]

    count_B_less_K = np.sum(B[:, 1::2] < K)
    sum_B_even_rows = np.sum(B[::2, :])

    if count_B_less_K > sum_B_even_rows:
     
        F[:N//2, :N//2] = np.flipud(C)
        F[N//2:, N//2:] = np.flipud(E)
    else:
        #
        F[:N//2, N//2:] = E
        F[:N//2, :N//2] = B

    return F

def compute_operations(A, F, K):
    det_A = np.linalg.det(A)
    sum_diag_F = np.trace(F)

    if det_A > sum_diag_F:
        result = np.linalg.inv(A) @ A.T - K * F
    else:
        G = np.tril(A)
        result = (np.linalg.inv(A) + G - F.T) * K

    return result

# Построение графиков
def plot_matrices(A, F, result):
    plt.figure(figsize=(15, 5))

    # График 1: Матрица A
    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap='viridis')
    plt.title('Matrix A')
    plt.colorbar()

    # График 2: Матрица F
    plt.subplot(1, 3, 2)
    plt.imshow(F, cmap='viridis')
    plt.title('Matrix F')
    plt.colorbar()

    # График 3: Результат матричных операций
    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='viridis')
    plt.title('Result of Matrix Operations')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Основная функция
def main():
    K = int(input("Введите число K: "))
    N = int(input("Введите размер матрицы N: "))

    A = generate_matrix(N)
    print("\nМатрица A:\n", A)
    plot_matrices(A, A, A)  

    F = form_matrix_F(A, K)
    print("\nМатрица F:\n", F)
    plot_matrices(A, F, F)  

    result = compute_operations(A, F, K)
    print("\nРезультат матричных операций:\n", result)
    plot_matrices(A, F, result) 

if __name__ == "__main__":
    main()


