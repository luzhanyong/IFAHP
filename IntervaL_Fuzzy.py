# 将Saaty’s scale标准转化为模糊区间
from decimal import Decimal, ROUND_HALF_UP
from utils import matrix_util
import numpy
import numpy as np



#定义强度矩阵
class Compare_Node():
    def __init__(self,com_degree,cert_defree):
        self.com_degree = com_degree
        self.cert_defree = cert_defree
    def print_info(self):
        print(f'{self.com_degree} ; {self.cert_defree}')





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

def scale_expert_degree_2(scale,expert_degree):
    value = scale * expert_degree * expert_degree
    if value < 1:
        return 1
    if value >=1 and value <= scale:
        return value

def scale_expert_degree(scale,expert_degree):
    value = scale * expert_degree
    if value < 1:
        return 1
    if value >=1 and value <= scale:
        return value


def re_scale_expert_degree_2(x, expert_degree):
    value = 1 / (x * expert_degree * expert_degree)
    if value > 1:
        return 1
    if value >= 1/x and value <= 1:
        return value


def re_scale_expert_degree(x, expert_degree):
    value = 1 / (x * expert_degree)
    if value > 1:
        return 1
    if value >= 1/x and value <= 1:
        return value


def round2(num):
    number = Decimal(num)
    rounded_number = number.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return rounded_number
class Interval:
    def __init__(self):
        self.L_u = 0
        self.L_l = 0
        self.m = 0
        self.U_l = 0
        self.U_u = 0


    def compute_Interval_fuzzy(self,scale,expert_degree):

        if scale == 1:
            self.L_u = 1
            self.L_l = 1
            self.m = 1
            self.U_l = 1
            self.U_u = 1
        elif scale in [2,3,4,5,6,7,8,9]:
            x = scale
            self.L_u = scale_expert_degree_2(scale,expert_degree)
            self.L_l = scale_expert_degree(scale,expert_degree)
            self.m = scale
            self.U_l = (2 - expert_degree) * scale
            self.U_u = (2 - expert_degree * expert_degree) * scale
        elif scale in [1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9]:
            x = 1 / scale
            self.L_u = 1 / ((2 - expert_degree * expert_degree) * x)
            self.L_l = 1 / ((2 - expert_degree) * x)
            self.m = scale
            self.U_l = re_scale_expert_degree(x,expert_degree)
            self.U_u = re_scale_expert_degree_2(x,expert_degree)
        else:
            print("Invalid scale")

        self.L_u = float(round2(self.L_u))
        self.L_l = float(round2(self.L_l))
        self.m = float(round2(self.m))
        self.U_l = float(round2(self.U_l))
        self.U_u = float(round2(self.U_u))


    def print_info(self):
        print(f'{self.L_u} ; {self.L_l} ; {self.m} ; {self.U_l} ; {self.U_u}')



#分列模糊矩阵
def fuzzy_number_matrix(data):


    # 初始化五个二维矩阵
    row = len(data)  # 每个二维矩阵中列表的长度

    matrix_L_u = numpy.empty([row,row], dtype=float)
    matrix_L_l = numpy.empty([row,row], dtype=float)
    matrix_m = numpy.empty([row,row], dtype=float)
    matrix_U_l = numpy.empty([row,row], dtype=float)
    matrix_U_u = numpy.empty([row,row], dtype=float)

    m = 0
    for i in range(len(data)):
        for j in range(len(data[i])):

            # if m % 4 == 0:
            #     print("\n")
            test_interval = Interval()
            test_interval.compute_Interval_fuzzy(data[i][j][0],data[i][j][1])
            # test_interval.print_info()
            m = m + 1

            matrix_L_u[i][j]= test_interval.L_u
            matrix_L_l[i][j]= test_interval.L_l
            matrix_m[i][j]=test_interval.m
            matrix_U_l[i][j]=test_interval.U_l
            matrix_U_u[i][j]=test_interval.U_u


    # print_matrix(matrix_L_u)
    # print_matrix(matrix_L_l)
    # print_matrix(matrix_m)
    # print_matrix(matrix_U_l)
    # print_matrix(matrix_U_u)

    # 转化为np类型
    # matrix_L_u = np.array(matrix_L_u)
    # matrix_L_l = np.array(matrix_L_l)
    # matrix_m = np.array(matrix_m)
    # matrix_U_l = np.array(matrix_U_l)
    # matrix_U_u = np.array(matrix_U_u)

    return matrix_L_u,matrix_L_l,matrix_m,matrix_U_l,matrix_U_u





def fuzzy_number_matrix_and_defuzzy(data1):


    # 初始化五个二维矩阵
    row = len(data1)  # 每个二维矩阵中列表的长度

    matrix_L_u = numpy.empty([row,row], dtype=float)
    matrix_L_l = numpy.empty([row,row], dtype=float)
    matrix_m = numpy.empty([row,row], dtype=float)
    matrix_U_l = numpy.empty([row,row], dtype=float)
    matrix_U_u = numpy.empty([row,row], dtype=float)

    m = 0

    result = numpy.empty([row,row], dtype=float)
    for i in range(len(data1)):
        for j in range(len(data1[i])):

            #第一个专家
            test_interval1 = Interval()
            test_interval1.compute_Interval_fuzzy(data1[i][j][0],data1[i][j][1])


            w = matrix_util.de_fuzzy(test_interval1.L_u,test_interval1.L_l,test_interval1.m,test_interval1.U_l,test_interval1.U_u,0.7)

            result[i][j] = w




    return result

def fuzzy_number_matrix_and_defuzzy2(data1,data2):


    # 初始化五个二维矩阵
    row = len(data1)  # 每个二维矩阵中列表的长度

    matrix_L_u = numpy.empty([row,row], dtype=float)
    matrix_L_l = numpy.empty([row,row], dtype=float)
    matrix_m = numpy.empty([row,row], dtype=float)
    matrix_U_l = numpy.empty([row,row], dtype=float)
    matrix_U_u = numpy.empty([row,row], dtype=float)

    m = 0

    result = numpy.empty([row,row], dtype=float)
    for i in range(len(data1)):
        for j in range(len(data1[i])):

            #第一个专家
            test_interval1 = Interval()
            test_interval1.compute_Interval_fuzzy(data1[i][j][0],data1[i][j][1])

            #第二个专家
            test_interval2 = Interval()
            test_interval2.compute_Interval_fuzzy(data2[i][j][0], data2[i][j][1])
            # test_interval.print_info()
            m = m + 1
            #聚合两个专家
            if(test_interval1.L_u > test_interval2.L_u):
                test_interval1.L_u = test_interval2.L_u
            if(test_interval1.L_l < test_interval2.L_l):
                test_interval1.L_l = test_interval2.L_l
            if(test_interval1.U_l > test_interval2.U_l):
                test_interval1.U_l = test_interval2.U_l
            if(test_interval1.U_u < test_interval2.U_u):
                test_interval1.U_u = test_interval2.U_u


            w = matrix_util.de_fuzzy(test_interval1.L_u,test_interval1.L_l,test_interval1.m,test_interval1.U_l,test_interval1.U_u,0.7)

            result[i][j] = w




    return result


if __name__ == '__main__':

    criteria = np.array([[1, 3, 5],
                         [1 / 3, 1, 2],
                         [1 / 5, 1 / 2, 1]])
    matrix_L_u,matrix_L_l,matrix_m,matrix_U_l,matrix_U_u = fuzzy_number_matrix(criteria)
    print(matrix_L_u)

    width, height = matrix_L_u.shape
    print(width)
    print(height)

    # 初始化五个二维矩阵
    num_lists = len(criteria)  # 每个二维矩阵中列表的长度

    matrix_L_u = [[] for _ in range(num_lists)]

    for i in range(len(criteria)):
        for j in range(len(criteria[i])):

            # if m % 4 == 0:
            #     print("\n")
            test_interval = Interval()
            test_interval.compute_Interval_fuzzy(criteria[i][j][0],criteria[i][j][1])
            # test_interval.print_info()
            m = m + 1

            matrix_L_u[i].append(test_interval.L_u)
            matrix_L_l[i].append(test_interval.L_l)
            matrix_m[i].append(test_interval.m)
            matrix_U_l[i].append(test_interval.U_l)
            matrix_U_u[i].append(test_interval.U_u)

        print_matrix(matrix_L_u)
        print_matrix(matrix_L_l)
        print_matrix(matrix_m)
        print_matrix(matrix_U_l)
        print_matrix(matrix_U_u)










