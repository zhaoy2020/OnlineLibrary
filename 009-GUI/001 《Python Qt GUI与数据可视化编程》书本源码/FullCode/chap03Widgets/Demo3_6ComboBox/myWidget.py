import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot

from PyQt5.QtGui import  QIcon

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
        
##  ==========由connectSlotsByName() 自动连接的槽函数====================        
   def on_btnIniItems_clicked(self):   ##“初始化列表”按钮
      #设置图标的操作
      icon=QIcon(":/icons/images/aim.ico")
      self.ui.comboBox.clear()    #清除列表
      provinces=["山东","河北","河南","湖北","湖南","广东"]    #列表数据
      for i in range(len(provinces)):
         self.ui.comboBox.addItem(icon,provinces[i])

##      #不设置图标的操作
##      self.ui.comboBox.clear()    #清除列表
##      provinces=["山东","河北","河南","湖北","湖南","广东"]  #列表数据
##      self.ui.comboBox.addItems(provinces)    #直接添加列表，但无法加图标


   def on_btnClearItems_clicked(self):    ##“清除列表”按钮
      self.ui.comboBox.clear()

   @pyqtSlot(bool)   ##“可编辑” CheckBox
   def on_chkBoxEditable_clicked(self,checked):  
      self.ui.comboBox.setEditable(checked)

   @pyqtSlot(str)    ##“简单的ComboBox”的当前项变化
   def on_comboBox_currentIndexChanged(self,curText):
      self.ui.lineEdit.setText(curText)

   def on_btnIni2_clicked(self):   ##有用户数据的comboBox2的初始化
      icon=QIcon(":/icons/images/unit.ico")
      self.ui.comboBox2.clear()
      cities={"北京":10, "上海":21, "天津":22,
              "徐州":516, "福州":591, "青岛":532}        #字典数据
      for k in cities:
         self.ui.comboBox2.addItem(icon,k,cities[k])

   @pyqtSlot(str)    ##当前项变化
   def on_comboBox2_currentIndexChanged(self,curText): 
      self.ui.lineEdit.setText(curText)
      zone=self.ui.comboBox2.currentData()    #读取关联数据
      if (zone != None):      #必须加此判断，因为有可能是None
         self.ui.lineEdit.setText(curText+":区号=%d"%zone)
        
##  =========自定义槽函数===================================        

   
##  ===========窗体测试程序 ================================        
if  __name__ == "__main__":         
   app = QApplication(sys.argv)   
   form=QmyWidget()      
   form.show()
   sys.exit(app.exec_())
