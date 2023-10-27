import sys

from PyQt5.QtWidgets import  QApplication, QWidget

##from PyQt5.QtCore import  pyqtSlot

from PyQt5.QtCore import   QTime, QTimer

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent) #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()      #创建UI对象
      self.ui.setupUi(self)    #构造UI界面

      self.timer=QTimer()     #创建定时器
      self.timer.stop()       #停止
      self.timer.setInterval(1000)    #定时周期1000ms
      self.timer.timeout.connect(self.do_timer_timeout)

      self.counter=QTime()    #创建计时器
        
##  =========由connectSlotsByName() 自动连接的槽函数============= 
   def on_btnStart_clicked(self):   ##“开始”按钮
      self.timer.start()    #开始定时
      self.counter.start()  #开始计时

      self.ui.btnStart.setEnabled(False)
      self.ui.btnStop.setEnabled(True)
      self.ui.btnSetIntv.setEnabled(False)

   def on_btnSetIntv_clicked(self):    ##设置定时器的周期  
      self.timer.setInterval(self.ui.spinBoxIntv.value())

   def on_btnStop_clicked(self):    ##“停止”按钮
      self.timer.stop()           #定时器停止
      tmMs=self.counter.elapsed() #计时器流逝的毫秒数

      ms=tmMs % 1000      #取余数，毫秒
      sec=tmMs/1000       #整秒
      timeStr="流逝的时间：%d 秒，%d 毫秒"%(sec, ms)
      self.ui.LabElapsedTime.setText(timeStr)

      self.ui.btnStart.setEnabled(True)
      self.ui.btnStop.setEnabled(False)
      self.ui.btnSetIntv.setEnabled(True)
        
        
##  ==========自定义槽函数==================================        
   def do_timer_timeout(self):     ##定时中断响应
      curTime=QTime.currentTime()      #获取当前时间
      self.ui.LCDHour.display(curTime.hour())
      self.ui.LCDMin.display(curTime.minute())
      self.ui.LCDSec.display(curTime.second())

   
##  =========窗体测试程序===================================        
if  __name__ == "__main__":       
   app = QApplication(sys.argv)  
   form=QmyWidget()         
   form.show()
   sys.exit(app.exec_())
