import time
import matplotlib.pyplot as plt
from IPython import display


class MyTimer():
    '''
    一个计时器，使用如下：
    timer = MyTimer()
    for i in range(3):
        time.sleep(0.01)
    timer()
    '''
    def __init__(self):
        '''初始化时候自动执行'''
        self.start = time.time()

    def __call__(self):
        '''再次调用该对象时，会自动执行'''
        self.stop = time.time()
        seconds = self.stop - self.start
        days = seconds // (24 * 3600)
        hours = (seconds % (24 * 3600)) // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        print('='*100, '\n', f"Total：\n {days} d \n {hours} h \n {minutes} m \n {remaining_seconds} s")
        
# class TrainPlot():
#     plt.close()
#     fig = plt.figure()

#     display.display(fig)
#     display.clear_output(wait=True)
def train_plot(data) -> None:
    '''
    在IPython中刷新的方式展示plot
    '''
    plt.close()
    fig = plt.figure()

    plt.plot(data)

    display.display(fig)
    display.clear_output(wait=True)