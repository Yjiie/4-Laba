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
                A[i, j] = np.random.randint(-10, 10)
            elif i >= N // 2 and j >= N // 2:
                A[i, j] = np.random.randint(-10, 10)
            elif i < N // 2 and j >= N // 2:
                A[i, j] = np.random.randint(-10, 10)
            else:
                A[i, j] = np.random.randint(-10, 10)
    return A

def form_matrix_F(A, K):
    N = A.shape[0]
    F = A.copy()
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

def print_matrix(matrix, name="Matrix"):
    formatted_matrix = np.array2string(
        matrix, 
        formatter={'float_kind': lambda x: f"{x:8.2f}"},
        max_line_width=100
    )
    print(f"\n{name}:\n{formatted_matrix}")

def plot_matrices(A, F, result):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(A, cmap='viridis')
    plt.title('Matrix A (Heatmap)')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    unique, counts = np.unique(F > 0, return_counts=True)
    labels = ['Positive', 'Negative or Zero']
    sizes = [counts[1], counts[0]]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FFC107'])
    plt.title('Matrix F (Positive Elements)')
    plt.subplot(2, 2, 3)
    plt.hist(result.flatten(), bins=20, color='skyblue', edgecolor='black')
    plt.title('Result Matrix (Histogram)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def main():
    K = int(input("Введите число K: "))
    N = int(input("Введите размер матрицы N (четное число): "))
    if N % 2 != 0:
        print("Ошибка: размер матрицы N должен быть четным!")
        return

    A = generate_matrix(N)
    print_matrix(A, name="Матрица A")
    F = form_matrix_F(A, K)
    print_matrix(F, name="Матрица F")
    result = compute_operations(A, F, K)
    print_matrix(result, name="Результат матричных операций")
    plot_matrices(A, F, result)

if __name__ == "__main__":
    main()



