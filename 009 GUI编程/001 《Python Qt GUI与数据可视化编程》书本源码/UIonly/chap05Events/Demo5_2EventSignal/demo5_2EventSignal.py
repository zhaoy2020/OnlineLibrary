import sys

from PyQt5.QtWidgets import  QApplication, QWidget,QLabel

from PyQt5.QtCore import  pyqtSignal

from PyQt5.QtGui import    QMouseEvent, QFont


## 自定义标签类，自定义了doubleClicked()信号
class QmyLabel(QLabel): 
   doubleClicked = pyqtSignal()  ##自定义信号
   
   def mouseDoubleClickEvent(self,event):    ##双击事件的处理
      self.doubleClicked.emit()
      

class QmyWidget(QWidget):
   def __init__(self, parent=None):
      super().__init__(parent) 
      self.resize(280,150)          
      self.setWindowTitle("Demo5_2,事件与信号")

      LabHello = QmyLabel(self) 
      LabHello.setText("双击我啊")

      font = LabHello.font()
      font.setPointSize(14)  
      font.setBold(True)     
      LabHello.setFont(font) 
      size=LabHello.sizeHint() 
      LabHello.setGeometry(70, 60, size.width(), size.height())
     
      LabHello.doubleClicked.connect(self.do_doubleClicked)

   def do_doubleClicked(self):  ##标签的槽函数
      print("标签被双击了")

   def mouseDoubleClickEvent(self,event): ##双击事件的处理
      print("窗口双击事件响应")
      
   
   
##  ============窗体测试程序 =================
if  __name__ == "__main__":  
   app = QApplication(sys.argv) 
   form=QmyWidget() 
   form.show()
   sys.exit(app.exec_())
