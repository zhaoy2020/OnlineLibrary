import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import  

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

##  =================自定义功能函数=================================

##  ==========由connectSlotsByName() 自动连接的槽函数===============        
   def on_btnCalculate_clicked(self):  ##"计算总价"按钮
      num=int(self.ui.editCount.text())
      price=float(self.ui.editPrice.text())
      total=num*price
      self.ui.editTotal.setText("%.2f" %total)

   @pyqtSlot(int)    ##"数量"SpinBox
   def on_spinCount_valueChanged(self,count):
      price=self.ui.spinPrice.value()
      self.ui.spinTotal.setValue(count*price)

   @pyqtSlot(float)     ##"单价" DoubleSpinBox
   def on_spinPrice_valueChanged(self,price):
      count=self.ui.spinCount.value()
      self.ui.spinTotal.setValue(count*price)

##  =============自定义槽函数===============================        
       

##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
