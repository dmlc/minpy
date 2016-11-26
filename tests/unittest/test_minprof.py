from minpy.utils.minprof import minprof
from inspect import getframeinfo, stack

def test_minprof():
    @minprof
    def recur_func(i):
        if i == 0:
            #print "did run"
            return
        return recur_func(i-1)
    
    recur_func(100)
    
    @minprof
    def fun_A():
        for i in range(2):
            fun_B()
    
    @minprof
    def fun_B():
        for i in range(3):
            fun_C()
    
    @minprof
    def fun_C():
        k = 0
        for i in range(4):
            fun_D()
            k += 1
    
    @minprof
    def fun_D():
        k = 0
        for i in range(5):
            k += 1
    
    with minprof():
        k = 0
        with minprof():
            k += 1
     
    fun_A()
    minprof.print_stats()

if __name__ == "__main__":
    test_minprof()
