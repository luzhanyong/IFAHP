
#打印矩型
def print_matrix(matrix):
    print('[', end=' ')
    for i in range(len(matrix)):
        print('[',end=' ')
        for j in range(len(matrix[i])):
            print(matrix[i][j],end=' ')
        if i + 1 == len(matrix):
            print(']', end=' ')
        else:
            print('],', end=' ')
    print(']')


#去模糊化
def de_fuzzy(L_u,L_l,m,U_l,U_u,k):


    # k = Decimal(k)
    up = (((U_u - L_u) + (m - L_u)) / 3) + L_u + (k * ((((U_l - L_l) + (m - L_l)) / 3) + L_l))
    W = up / 2
    return W


if __name__ == '__main__':
    import numpy as np

    # 假设这是你的数据列表
    data = [4, 2, 6, 1, 8]

    # 将数据转换为 NumPy 数组
    data_np = np.array(data)

    # 使用 argsort 函数获取按照元素大小排序后的索引顺序
    sorted_indexes = np.argsort(data_np)

    # 输出排序后的索引顺序
    print(sorted_indexes)




