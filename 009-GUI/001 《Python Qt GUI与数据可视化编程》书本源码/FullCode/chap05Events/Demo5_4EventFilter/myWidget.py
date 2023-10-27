import sys

from PyQt5.QtWidgets import  QApplication,  QWidget, qApp

from PyQt5.QtCore import  Qt,QEvent

##from PyQt5.QtGui import  QKeyEvent

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.ui.LabHover.installEventFilter(self)
      self.ui.LabDBClick.installEventFilter(self)
      qApp.processEvents()


##  =================自定义功能函数=================================
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        

##  ==================event处理函数=================================
   def eventFilter(self,watched,event):
      if (watched==self.ui.LabHover):     #上面的QLabel组件
         if (event.type()==QEvent.Enter): #鼠标光标进入
            self.ui.LabHover.setStyleSheet("background-color: rgb(170, 255, 255);")
         elif (event.type()==QEvent.Leave):     #鼠标光标移出
            self.ui.LabHover.setStyleSheet("")
            self.ui.LabHover.setText("靠近我，点击我")
         elif (event.type()==QEvent.MouseButtonPress):   #鼠标按键按下
            self.ui.LabHover.setText("button pressed")
         elif (event.type()==QEvent.MouseButtonRelease): #鼠标按键释放
            self.ui.LabHover.setText("button released")

      if (watched==self.ui.LabDBClick):   #下面的QLabel组件
         if (event.type()==QEvent.Enter):
            self.ui.LabDBClick.setStyleSheet("background-color: rgb(85, 255, 127);")
         elif (event.type()==QEvent.Leave):
            self.ui.LabDBClick.setStyleSheet("") 
            self.ui.LabDBClick.setText("可双击的标签")
         elif (event.type()==QEvent.MouseButtonDblClick):   #鼠标双击
            self.ui.LabDBClick.setText("double clicked")

      return super().eventFilter(watched,event)
##      return True
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":     
   app = QApplication(sys.argv)
   form=QmyWidget()            
   form.show()
   sys.exit(app.exec_())
