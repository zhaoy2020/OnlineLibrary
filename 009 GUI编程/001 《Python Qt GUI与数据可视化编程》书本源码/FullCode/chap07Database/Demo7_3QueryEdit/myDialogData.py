import sys

from PyQt5.QtWidgets import  QApplication, QDialog,QFileDialog

##from PyQt5.QtWidgets import  

from PyQt5.QtCore import  pyqtSlot, Qt, QFile,QIODevice,QDate

from PyQt5.QtSql import   QSqlRecord  

##from PyQt5.QtCore import  pyqtSlot,Qt

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import QPixmap

from ui_QWDialogData import Ui_QWDialogData

class QmyDialogData(QDialog): 
   def __init__(self,parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_QWDialogData()  #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.__record=QSqlRecord() #用于存储一条记录的数据
      
##  ==============自定义功能函数============
   def setInsertRecord(self,recData): ##设置插入记录的数据
      self.__record=recData
      self.ui.spinEmpNo.setEnabled(True)  #员工编号允许编辑
      self.setWindowTitle("插入新记录")
      self.ui.spinEmpNo.setValue(recData.value("empNo"))
      

   def setUpdateRecord(self,recData): ##设置更新记录的数据
      self.__record=recData
      self.ui.spinEmpNo.setEnabled(False)    #员工编号不允许编辑
      self.setWindowTitle("更新记录")

##根据recData的数据更新界面显示
      self.ui.spinEmpNo.setValue(recData.value("empNo"))
      self.ui.editName.setText(recData.value("Name"))
      self.ui.comboSex.setCurrentText(recData.value("Gender"))

      birth=recData.value("Birthday")     #str类型, 注意！！！
      birth_date=QDate.fromString(birth,"yyyy-MM-dd")
      self.ui.editBirth.setDate(birth_date)

      self.ui.comboProvince.setCurrentText(recData.value("Province"))
      self.ui.comboDep.setCurrentText(recData.value("Department"))
      self.ui.spinSalary.setValue(recData.value("Salary"))
      self.ui.editMemo.setPlainText(recData.value("Memo"))

      picData=recData.value("Photo")  #bytearray类型
      if (picData==''):  #图片字段内容为空
         self.ui.LabPhoto.clear()
      else:
         pic=QPixmap()
         pic.loadFromData(picData)
         W=self.ui.LabPhoto.size().width()
         self.ui.LabPhoto.setPixmap(pic.scaledToWidth(W))

   def getRecordData(self): ##返回界面编辑的数据
      self.__record.setValue("empNo",  self.ui.spinEmpNo.value())
      self.__record.setValue("Name",   self.ui.editName.text())
      self.__record.setValue("Gender", self.ui.comboSex.currentText())
      self.__record.setValue("Birthday",  self.ui.editBirth.date())

      self.__record.setValue("Province",  self.ui.comboProvince.currentText())
      self.__record.setValue("Department",self.ui.comboDep.currentText())

      self.__record.setValue("Salary", self.ui.spinSalary.value())
      self.__record.setValue("Memo",   self.ui.editMemo.toPlainText())
##照片编辑时已经修改了self.__record的photo字段的值

      return  self.__record  #以记录作为返回值
      
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“导入照片”按钮
   def on_btnSetPhoto_clicked(self):
      fileName,filt=QFileDialog.getOpenFileName(self,"选择图片","","照片(*.jpg)")
      if (fileName==''):
       return

      file=QFile(fileName)    
      file.open(QIODevice.ReadOnly)
      data = file.readAll()
      file.close()

      self.__record.setValue("Photo",data)   #图片保存到Photo字段

      pic=QPixmap()
      pic.loadFromData(data)  #在界面上显示
      W=self.ui.LabPhoto.width()
      self.ui.LabPhoto.setPixmap(pic.scaledToWidth(W))


   @pyqtSlot()    ##“清除照片按钮”
   def on_btnClearPhoto_clicked(self):
      self.ui.LabPhoto.clear()
      self.__record.setNull("Photo")   #Photo字段清空
      
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyDialogData()            #创建窗体
   form.show()
   sys.exit(app.exec_())
