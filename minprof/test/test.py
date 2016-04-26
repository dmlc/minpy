def fun_A():
    k = 0
    for i in range(100):
        k = k + 1
        fun_B()

@profile
def fun_B():
    k = 0
    for i in range(100):
        k = k + 1
        fun_C()

def fun_C():
    k = 0
    for i in range(100):
        k = k + 1


fun_A()
