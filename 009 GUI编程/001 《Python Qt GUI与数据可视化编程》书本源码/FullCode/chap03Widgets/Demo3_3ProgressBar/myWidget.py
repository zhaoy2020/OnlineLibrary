import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.ui.slider.setMaximum(200)
      self.ui.scrollBar.setMaximum(200)
      self.ui.progressBar.setMaximum(200)

      self.ui.slider.valueChanged.connect(self.do_valueChanged)
      self.ui.scrollBar.valueChanged.connect(self.do_valueChanged)

        
##  ========由connectSlotsByName() 自动连接的槽函数=================        
   def on_radio_Percent_clicked(self):     ##“显示格式--百分比”
      self.ui.progressBar.setFormat("%p%")

   def on_radio_Value_clicked(self):       ##“显示格式--当前值”
      self.ui.progressBar.setFormat("%v")
        
   @pyqtSlot(bool)   ##“textVisible”复选框
   def on_chkBox_Visible_clicked(self,checked):    
      self.ui.progressBar.setTextVisible(checked)
        
   @pyqtSlot(bool)  ##"InvertedAppearance"复选框
   def on_chkBox_Inverted_clicked(self,checked):  
      self.ui.progressBar.setInvertedAppearance(checked)


##  ========自定义槽函数====================================        
   def do_valueChanged(self,value):   ##slider和scrollBar的valueChanged信号的响应
      self.ui.progressBar.setValue(value)

   
##  ===========窗体测试程序 =================================        
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
