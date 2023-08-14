import sys

from PyQt5.QtWidgets import  (QApplication, QDialog,QFileDialog,
                        QColorDialog,QFontDialog,QProgressDialog,
                        QLineEdit,QInputDialog,QMessageBox)

from PyQt5.QtCore import  Qt, pyqtSlot, QDir,QTime

from PyQt5.QtGui import QPalette, QColor, QFont

from ui_Dialog import Ui_Dialog


class QmyDialog(QDialog): 
   def __init__(self, parent=None):
      super().__init__(parent)      #调用父类构造函数，创建窗体
      self.ui=Ui_Dialog()           #创建UI对象
      self.ui.setupUi(self)         #构造UI界面

##  ==============自定义功能函数============
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()   ##"打开一个文件"
   def on_btnOpen_clicked(self):   
      curPath=QDir.currentPath()    #获取系统当前目录
      dlgTitle="选择一个文件"       #对话框标题
      filt="所有文件(*.*);;文本文件(*.txt);;图片文件(*.jpg *.gif *.png)"   #文件过滤器

      filename,filtUsed=QFileDialog.getOpenFileName(self,dlgTitle,curPath,filt)
      self.ui.plainTextEdit.appendPlainText(filename)
      self.ui.plainTextEdit.appendPlainText("\n"+filtUsed)
            
   @pyqtSlot()    ##"打开多个文件"
   def on_btnOpenMulti_clicked(self):   
      curPath=QDir.currentPath()    #获取系统当前目录
      dlgTitle="选择一个文件"       #对话框标题
      filt="所有文件(*.*);;文本文件(*.txt);;图片文件(*.jpg *.gif *.png)"   #文件过滤器

      fileList,filtUsed=QFileDialog.getOpenFileNames(self,dlgTitle,curPath,filt)
      for i in range(len(fileList)):
         self.ui.plainTextEdit.appendPlainText(fileList[i])

      self.ui.plainTextEdit.appendPlainText("\n"+filtUsed)

   @pyqtSlot()    ##“选择已有目录 ”
   def on_btnSelDir_clicked(self): 
      curPath=QDir.currentPath()    #获取系统当前目录
      dlgTitle="选择一个目录"       #对话框标题
      selectedDir=QFileDialog.getExistingDirectory(self,
               dlgTitle,curPath,QFileDialog.ShowDirsOnly)
      self.ui.plainTextEdit.appendPlainText("\n"+selectedDir)

   @pyqtSlot()    ##“保存文件”
   def on_btnSave_clicked(self):   
      curPath=QDir.currentPath()    #获取系统当前目录
      dlgTitle="保存文件"   #对话框标题
      filt="所有文件(*.*);;文本文件(*.txt);;图片文件(*.jpg *.gif *.png)" #文件过滤器

      filename,filtUsed=QFileDialog.getSaveFileName(self,dlgTitle,curPath,filt)
      self.ui.plainTextEdit.appendPlainText(filename)
      self.ui.plainTextEdit.appendPlainText("\n"+filtUsed)
        
   @pyqtSlot()    ##"选择颜色"
   def on_btnColor_clicked(self): 
      pal=self.ui.plainTextEdit.palette() #获取现有 palette
      iniColor=pal.color(QPalette.Text)   #现有的文字颜色
      color=QColorDialog.getColor(iniColor,self,"选择颜色")
      if color.isValid():    #选择有效
         pal.setColor(QPalette.Text,color)       #palette 设置选择的颜色
         self.ui.plainTextEdit.setPalette(pal)   #设置 palette
        
   @pyqtSlot()    ##"选择字体"
   def on_btnFont_clicked(self):   
      iniFont=self.ui.plainTextEdit.font() #获取文本框的字体
      font,OK=QFontDialog.getFont(iniFont) #选择字体, 注意与C++版本不同
      if (OK):     #选择有效
         self.ui.plainTextEdit.setFont(font)


   @pyqtSlot()    ##"进度对话框"
   def on_btnProgress_clicked(self): 
      labText="正在复制文件..."   #文字信息
      btnText="取消"      #"取消"按钮的标题
      minV=0
      maxV=200
      dlgProgress=QProgressDialog(labText,btnText, minV, maxV, self)
      dlgProgress.canceled.connect(self.do_progress_canceled) #canceled信号

      dlgProgress.setWindowTitle("复制文件")
      dlgProgress.setWindowModality(Qt.WindowModal)   #模态对话框

      dlgProgress.setAutoReset(True)  #calls reset() as soon as value() equals maximum()
      dlgProgress.setAutoClose(True)  #whether the dialog gets hidden by reset()

      msCounter=QTime() #计时器
      for i in range(minV,maxV+1):
         dlgProgress.setValue(i)
         dlgProgress.setLabelText("正在复制文件,第 %d 个"%i)

         msCounter.start() #计时器重新开始
         while(msCounter.elapsed()<30):  #延时 30ms
            None

         if (dlgProgress.wasCanceled()): #中途取消
            break


   @pyqtSlot()    ##“输入字符串”
   def on_btnInputString_clicked(self):    
      dlgTitle="输入文字对话框"
      txtLabel="请输入文件名"
      defaultInput="新建文件.txt"
      echoMode=QLineEdit.Normal     #正常文字输入
##    echoMode=QLineEdit.Password  #密码输入

      text,OK = QInputDialog.getText(self, dlgTitle,txtLabel,
                                    echoMode,defaultInput)
      if (OK):
         self.ui.plainTextEdit.appendPlainText(text)
        
        
   @pyqtSlot()    ##“输入整数”
   def on_btnInputInt_clicked(self):
      dlgTitle="输入整数对话框"
      txtLabel="设置字体大小"
      defaultValue=self.ui.plainTextEdit.font().pointSize()   #现有字体大小
      minValue=6
      maxValue=50
      stepValue=1

      inputValue,OK = QInputDialog.getInt(self, dlgTitle,txtLabel,
                            defaultValue, minValue, maxValue,stepValue)
      if OK: 
         font=self.ui.plainTextEdit.font()
         font.setPointSize(inputValue)
         self.ui.plainTextEdit.setFont(font)

   @pyqtSlot()    ##“输入浮点数”
   def on_btnInputFloat_clicked(self):
      dlgTitle="输入浮点数对话框"
      txtLabel="输入一个浮点数"
      defaultValue=3.65
      minValue=0
      maxValue=10000
      decimals=2

      inputValue,OK = QInputDialog.getDouble(self, dlgTitle,txtLabel,
                            defaultValue, minValue, maxValue,decimals)
      if OK:
         text="输入了一个浮点数：%.2f"%inputValue
         self.ui.plainTextEdit.appendPlainText(text)

   @pyqtSlot()    ##"条目选择输入"
   def on_btnInputItem_clicked(self):
      dlgTitle="条目选择对话框"
      txtLabel="请选择级别"
      curIndex=0
      editable=True

      items=["优秀","良好","合格","不合格"]

      text,OK = QInputDialog.getItem(self, dlgTitle,txtLabel,
                            items, curIndex, editable)
      if OK:
         self.ui.plainTextEdit.appendPlainText(text)
        
   @pyqtSlot()    ##"question"
   def on_btnMsgQuestion_clicked(self):
      dlgTitle="Question消息框"
      strInfo="文件已被修改，是否保存修改？"
      defaultBtn=QMessageBox.NoButton  #缺省按钮
      result=QMessageBox.question(self, dlgTitle, strInfo,
                   QMessageBox.Yes|QMessageBox.No |QMessageBox.Cancel,
                   defaultBtn)

      if (result==QMessageBox.Yes):
         self.ui.plainTextEdit.appendPlainText("Question消息框: Yes 被选择")
      elif(result==QMessageBox.No):
         self.ui.plainTextEdit.appendPlainText("Question消息框: No 被选择")
      elif(result==QMessageBox.Cancel):
         self.ui.plainTextEdit.appendPlainText("Question消息框: Cancel 被选择")
      else:
         self.ui.plainTextEdit.appendPlainText("Question消息框: 无选择")

   @pyqtSlot()    ##"information"
   def on_btnMsgInformation_clicked(self):
      dlgTitle="information消息框"
      strInfo="文件已经被正确打开."
      QMessageBox.information(self, dlgTitle, strInfo)
         

   @pyqtSlot()    ##"warning"
   def on_btnMsgWarning_clicked(self):
      dlgTitle="warning消息框"
      strInfo="文件内容已经被修改."
      QMessageBox.warning(self, dlgTitle, strInfo)

   @pyqtSlot()    ##"critical"
   def on_btnMsgCritical_clicked(self):
      dlgTitle="critical消息框"
      strInfo="出现严重错误，程序将关闭."
      QMessageBox.critical(self, dlgTitle, strInfo)

   @pyqtSlot()    ##"about"
   def on_btnMsgAbout_clicked(self):
      dlgTitle="about消息框"
      strInfo="Python Qt GUI与数据可视化编程\n保留所有版权"
      QMessageBox.about(self, dlgTitle, strInfo)

   @pyqtSlot()    ##"About Qt"
   def on_btnMsgAboutQt_clicked(self):
      dlgTitle="aboutQt消息框"
      QMessageBox.aboutQt(self, dlgTitle)

    
##  =============自定义槽函数===============================        
   def do_progress_canceled(self):  ##取消进度对话框
      self.ui.plainTextEdit.appendPlainText("**进度对话框被取消了**")
   
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":      
   app = QApplication(sys.argv) 
   form=QmyDialog()           
   form.show()
   sys.exit(app.exec_())
