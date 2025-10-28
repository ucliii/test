import time
def time_master(func):
    def call_func():
        print("开始运行程序...")
        start = time.time()
        func()
        stop = time.time()
        print("结束程序运行...")
        print(f"一共耗费了{(stop - start):.4f}秒")
    return call_func

def myfunc():
    i = 1
    while i <= 9 :
        j = 1
        while j <= i :
            print(i,"*",j,"=",i*j,end =" ")
            j += 1
        i += 1
        print()

myfunc = time_master(myfunc)

myfunc()
