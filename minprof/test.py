import time


@prof
def fun_A():
    k = 0
    for i in range(2):
        k = k + 1
        fun_B()

@prof
def fun_A1():
    k = 0
    for i in range(2):
        k = k + 1
        fun_B()

@prof
def fun_B():
    k = 0
    for i in range(3):
        k = k + 1
        fun_C()

@prof
def fun_C():
    k = 0
    for i in range(4):
        k = k + 1

fun_A()
fun_A1()
