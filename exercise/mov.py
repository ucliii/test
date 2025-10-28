origin = (0,0)          #这个是原点
legal_x = [-100,100]    #限定x轴的移动范围
legal_y = [-100,100]    #限定y轴的移动范围

def create(pos_x=0,pos_y=0):
    def moving(direction,step):
        nonlocal pos_x, pos_y
        new_x = pos_x + direction[0] * step
        new_y = pos_y + direction[1] * step

        if new_x < legal_x[0]:
            pos_x = legal_x[0] - (new_x - legal_x[0])
        elif new_x > legal_x[1]:
            pos_x = legal_x[1] - (new_x - legal_x[1])
        else:
            pos_x = new_x

        if new_y < legal_y[0]:
            pos_y = legal_y[0] - (new_y - legal_y[0])
        elif new_y > legal_y[1]:
            pos_y = legal_y[1] - (new_y - legal_y[1])
        else:
            pos_y = new_y
        
        return pos_x, pos_y
    return moving
   
move = create()
print("向右移动20步后，位置是：",move([1,0],20))
print("向上移动120步后，位置是：",move([0,1],120))