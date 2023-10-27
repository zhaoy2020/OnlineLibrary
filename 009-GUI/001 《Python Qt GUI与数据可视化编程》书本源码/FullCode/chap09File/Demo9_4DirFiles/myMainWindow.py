# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  (pyqtSlot, pyqtSignal, Qt, QFileSystemWatcher,
                        QCoreApplication, QDir, QFileInfo, QFile)

from PyQt5.QtWidgets import  QFileDialog,QMessageBox

##from PyQt5.QtGui import

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.ui.toolBox.setCurrentIndex(0)
      self.fileWatcher=QFileSystemWatcher()
      self.fileWatcher.directoryChanged.connect(self.do_directoryChanged)
      self.fileWatcher.fileChanged.connect(self.do_fileChanged)
      

##  ==============自定义功能函数========================
   def __showBtnInfo(self,btn):  ##显示按钮的text()和toolTip()
      self.ui.textEdit.appendPlainText("===="+btn.text())
      self.ui.textEdit.appendPlainText(btn.toolTip()+"\n")
      

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============
   @pyqtSlot()    ##"选择文件"按钮
   def on_btnOpenFile_clicked(self):
      curDir=QDir.currentPath() #获取当前路径
      aFile,filt=QFileDialog.getOpenFileName(self,
                           "打开文件",curDir,"所有文件(*.*)")
      self.ui.editFile.setText(aFile)
      
   @pyqtSlot()    ##"选择目录"按钮
   def on_btnOpenDir_clicked(self):
      curDir=QDir.currentPath()
      aDir=QFileDialog.getExistingDirectory(self,"选择一个目录",
                              curDir,QFileDialog.ShowDirsOnly)
      self.ui.editDir.setText(aDir)

   @pyqtSlot()    ##"清空"按钮
   def on_btnClear_clicked(self):
      self.ui.textEdit.clear()
      

## =========QFile类 的静态函数===========
   @pyqtSlot()    ##类函数copy()
   def on_btnFile_copy_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text().strip()   #源文件
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return
      
      fileInfo=QFileInfo(sous)
      newFile=fileInfo.path()+"/"+fileInfo.baseName()+"--副本."+fileInfo.suffix()
      if QFile.copy(sous,newFile):  
         self.ui.textEdit.appendPlainText("源文件："+sous)
         self.ui.textEdit.appendPlainText("复制为文件："+newFile+"\n")
      else:
         self.ui.textEdit.appendPlainText("复制文件失败")
      

   @pyqtSlot()  ##类函数exists()
   def on_btnFile_exists_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text().strip()#源文件
      if QFile.exists(sous):
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")
         
                                          
   @pyqtSlot()  ##类函数remove()
   def on_btnFile_remove_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text().strip()   #源文件
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return

      ret=QMessageBox.question(self,"确认删除","确定要删除这个文件吗\n\n"+sous)
      if (ret!=QMessageBox.Yes):
         return

      if QFile.remove(sous):
         self.ui.textEdit.appendPlainText("成功删除文件："+sous+"\n")
      else:
         self.ui.textEdit.appendPlainText("删除文件失败："+sous+"\n")

   @pyqtSlot()    ##类函数rename()
   def on_btnFile_rename_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text().strip()   #源文件
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return

      fileInfo=QFileInfo(sous)
      newFile=fileInfo.path()+"/"+fileInfo.baseName()+".XZY"  #更改文件后缀为".XYZ"
      if QFile.rename(sous,newFile):   
         self.ui.textEdit.appendPlainText("源文件："+sous)
         self.ui.textEdit.appendPlainText("重命名为："+newFile+"\n")
      else:
         self.ui.textEdit.appendPlainText("重命名文件失败\n")
      

## =========QFileInfo类===========
   @pyqtSlot()  ##absoluteFilePath()
   def on_btnInfo_absFilePath_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.absoluteFilePath()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##absolutePath()
   def on_btnInfo_absPath_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.absolutePath()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##fileName()
   def on_btnInfo_fileName_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.fileName()
      self.ui.textEdit.appendPlainText(text+"\n")


   @pyqtSlot()  ##filePath()
   def on_btnInfo_filePath_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.filePath()
      self.ui.textEdit.appendPlainText(text+"\n")


   @pyqtSlot()  ##size()
   def on_btnInfo_size_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      btCount=fileInfo.size()    #字节数
      text="%d Bytes"%btCount
      self.ui.textEdit.appendPlainText(text+"\n")


   @pyqtSlot()  ##path()
   def on_btnInfo_path_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.path()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##baseName()
   def on_btnInfo_baseName_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.baseName()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##completeBaseName()
   def on_btnInfo_baseName2_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.completeBaseName()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##suffix()
   def on_btnInfo_suffix_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.suffix()
      self.ui.textEdit.appendPlainText(text+"\n")
      
   @pyqtSlot()  ##completeSuffix()
   def on_btnInfo_suffix2_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      text=fileInfo.completeSuffix()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##isDir()
   def on_btnInfo_isDir_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editDir.text())
      if fileInfo.isDir():
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")

   @pyqtSlot()  ##isFile()
   def on_btnInfo_isFile_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      if fileInfo.isFile():
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")


   @pyqtSlot()  ##isExecutable()
   def on_btnInfo_isExec_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      if fileInfo.isExecutable():
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")

   @pyqtSlot()  ##birthTime() ,替代了过时的created()函数
   def on_btnInfo_birthTime_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      dt=fileInfo.birthTime()    # QDateTime
      text=dt.toString("yyyy-MM-dd hh:mm:ss")
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##lastModified()
   def on_btnInfo_lastModified_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      dt=fileInfo.lastModified()    # QDateTime
      text=dt.toString("yyyy-MM-dd hh:mm:ss")
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##lastRead()
   def on_btnInfo_lastRead_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      dt=fileInfo.lastRead()  # QDateTime
      text=dt.toString("yyyy-MM-dd hh:mm:ss")
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##类函数exists()
   def on_btnInfo_exists_clicked(self):
      self.__showBtnInfo(self.sender())
      
      if QFileInfo.exists(self.ui.editFile.text()):
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")

   @pyqtSlot()  ##接口函数exists()
   def on_btnInfo_exists2_clicked(self):
      self.__showBtnInfo(self.sender())
      fileInfo=QFileInfo(self.ui.editFile.text())
      if fileInfo.exists():
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")

## ==================QDir类========================
   @pyqtSlot()  ##tempPath()
   def on_btnDir_tempPath_clicked(self):
      self.__showBtnInfo(self.sender())
      text=QDir.tempPath()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##rootPath()
   def on_btnDir_rootPath_clicked(self):
      self.__showBtnInfo(self.sender())
      text=QDir.rootPath()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##homePath()
   def on_btnDir_homePath_clicked(self):
      self.__showBtnInfo(self.sender())
      text=QDir.homePath()
      self.ui.textEdit.appendPlainText(text+"\n")
         
   @pyqtSlot()  ##drives()
   def on_btnDir_drives_clicked(self):
      self.__showBtnInfo(self.sender())
      strList=QDir.drives()   #QFileInfoList
      for line in strList:    #line 是QFileInfo类型
         self.ui.textEdit.appendPlainText(line.path())
      self.ui.textEdit.appendPlainText("")

   @pyqtSlot()  ##currentPath()
   def on_btnDir_curPath_clicked(self):
      self.__showBtnInfo(self.sender())
      text=QDir.currentPath()
      self.ui.textEdit.appendPlainText(text+"\n")
      

   @pyqtSlot()  ##setCurrent()
   def on_btnDir_setCurPath_clicked(self):
      self.__showBtnInfo(self.sender())

      curDir=QDir.currentPath()
      text=QFileDialog.getExistingDirectory(self,
                     "选择一个目录", curDir,QFileDialog.ShowDirsOnly)
      QDir.setCurrent(text)
      self.ui.textEdit.appendPlainText("选择了一个目录作为当前目录：\n"+text+"\n")


   @pyqtSlot()  ##mkdir()
   def on_btnDir_mkdir_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text().strip()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return

      subDir="subdir1"
      dirObj=QDir(sous)
      if dirObj.mkdir(subDir):
         self.ui.textEdit.appendPlainText("新建一个子目录: "+subDir+"\n")
      else:
         self.ui.textEdit.appendPlainText("创建目录失败\n")
         

   @pyqtSlot()  ##rmdir()
   def on_btnDir_rmdir_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text().strip()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return

      dirObj=QDir(sous)
      if dirObj.rmdir(sous):
         self.ui.textEdit.appendPlainText("成功删除所选目录\n"+sous+"\n")
      else:
         self.ui.textEdit.appendPlainText("删除目录失败,目录下必须为空\n")

   @pyqtSlot()  ##remove()
   def on_btnDir_remove_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text().strip()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return
      
      parDir=self.ui.editDir.text().strip()
      if parDir=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return

      dirObj=QDir(parDir)
      if dirObj.remove(sous):
         self.ui.textEdit.appendPlainText("成功删除文件:\n"+sous+"\n")
      else:
         self.ui.textEdit.appendPlainText("删除文件失败\n")

   @pyqtSlot()  ##rename()
   def on_btnDir_rename_clicked(self):
      self.__showBtnInfo(self.sender())
      
      sous=self.ui.editFile.text()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return

      parDir=self.ui.editDir.text().strip()
      if parDir=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return
      
      dirObj=QDir(parDir)
      
      fileInfo=QFileInfo(sous)
      newFile=fileInfo.path()+"/"+fileInfo.baseName()+".XYZ"
      
      if dirObj.rename(sous,newFile):
         self.ui.textEdit.appendPlainText("源文件："+sous)
         self.ui.textEdit.appendPlainText("重命名为："+newFile+"\n")
      else:
         self.ui.textEdit.appendPlainText("重命名文件失败\n")
         

   @pyqtSlot()  ##setPath(),改换QDir所指的目录
   def on_btnDir_setPath_clicked(self):
      self.__showBtnInfo(self.sender())
      curDir=QDir.currentPath()
      lastDir=QDir(curDir)
      self.ui.textEdit.appendPlainText("选择目录之前,QDir所指目录是："+lastDir.absolutePath())

      aDir=QFileDialog.getExistingDirectory(self,"选择一个目录",curDir,QFileDialog.ShowDirsOnly)
      if aDir=="":
         return

      lastDir.setPath(aDir)
      self.ui.textEdit.appendPlainText("\n选择目录之后,QDir所指目录是："+lastDir.absolutePath()+"\n")
      

   @pyqtSlot()  ##removeRecursively()
   def on_btnDir_removeALL_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text().strip()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return

      dirObj=QDir(sous)
      ret=QMessageBox.question(self,"确认删除","确认删除目录下的所有文件及目录吗？\n"+sous)
      if ret !=QMessageBox.Yes:
         return
      
      if dirObj.removeRecursively():
         self.ui.textEdit.appendPlainText("删除目录及文件成功\n")
      else:
         self.ui.textEdit.appendPlainText("删除目录及文件失败\n")


   @pyqtSlot()  ##absoluteFilePath()
   def on_btnDir_absFilePath_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return

      parDir=QDir.currentPath()
      dirObj=QDir(parDir)
      text=dirObj.absoluteFilePath(sous)
      self.ui.textEdit.appendPlainText(text+"\n")
      

   @pyqtSlot()  ##absolutePath()
   def on_btnDir_absPath_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return

      dirObj=QDir(sous)
      text=dirObj.absolutePath()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##canonicalPath()
   def on_btnDir_canonPath_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个目录")
         return

      dirObj=QDir(sous)
      text=dirObj.canonicalPath()
      self.ui.textEdit.appendPlainText(text+"\n")

      
   @pyqtSlot()  ##filePath()
   def on_btnDir_filePath_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editFile.text()
      if sous=="":
         self.ui.textEdit.appendPlainText("请先选择一个文件")
         return

      parDir=QDir.currentPath()
      dirObj=QDir(parDir)
      text=dirObj.filePath(sous)
      self.ui.textEdit.appendPlainText(text+"\n")

      
   @pyqtSlot()  ##exists()
   def on_btnDir_exists_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text()
##      if sous=="":
##         self.ui.textEdit.appendPlainText("请先选择一个目录")
##         return

      dirObj=QDir(sous)  #若sous为空，则使用其当前目录
      self.ui.textEdit.appendPlainText(dirObj.absolutePath()+"\n")
      if dirObj.exists():
         self.ui.textEdit.appendPlainText("True \n")
      else:
         self.ui.textEdit.appendPlainText("False \n")


   @pyqtSlot()  ##dirName()
   def on_btnDir_dirName_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text()
##      if sous=="":
##         self.ui.textEdit.appendPlainText("请先选择一个目录")
##         return

      dirObj=QDir(sous)  #若sous为空，则使用其当前目录
      text=dirObj.dirName()
      self.ui.textEdit.appendPlainText(text+"\n")

   @pyqtSlot()  ##entryList()dirs
   def on_btnDir_listDir_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text()
      dirObj=QDir(sous)  #若sous为空，则使用其当前目录
      strList=dirObj.entryList(QDir.Dirs | QDir.NoDotAndDotDot)
      self.ui.textEdit.appendPlainText("所选目录下的所有目录:")
      for line in strList:
         self.ui.textEdit.appendPlainText(line)
      self.ui.textEdit.appendPlainText("\n")
      
   @pyqtSlot()  ##entryList()files
   def on_btnDir_listFile_clicked(self):
      self.__showBtnInfo(self.sender())
      sous=self.ui.editDir.text()
      dirObj=QDir(sous)  #若sous为空，则使用其当前目录
      strList=dirObj.entryList(QDir.Files)
      self.ui.textEdit.appendPlainText("所选目录下的所有文件:")
      for line in strList:
         self.ui.textEdit.appendPlainText(line)
      self.ui.textEdit.appendPlainText("\n")

## ==========QFileSystemWatcher类===================

   @pyqtSlot()  ##addPath()添加监听目录
   def on_btnWatch_addDir_clicked(self):
      self.__showBtnInfo(self.sender())
      curDir=QDir.currentPath()
      aDir=QFileDialog.getExistingDirectory(self,"选择一个需要监听的目录",
                     curDir,QFileDialog.ShowDirsOnly)
      self.fileWatcher.addPath(aDir)   #添加监听目录
      self.ui.textEdit.appendPlainText("添加的监听目录：")
      self.ui.textEdit.appendPlainText(aDir+"\n")


   @pyqtSlot()  ##addPaths()添加监听文件
   def on_btnWatch_addFiles_clicked(self):
      self.__showBtnInfo(self.sender())

      curDir=QDir.currentPath()
      fileList,flt = QFileDialog.getOpenFileNames(self,"选择需要监听的文件",
                            curDir, "所有文件 (*.*)")
      self.fileWatcher.addPaths(fileList)  #添加监听文件列表

      self.ui.textEdit.appendPlainText("添加的监听文件：")
      for lineStr in fileList:
         self.ui.textEdit.appendPlainText(lineStr)
      self.ui.textEdit.appendPlainText("")

   @pyqtSlot()  ##removePaths()移除所有监听的文件和目录
   def on_btnWatch_remove_clicked(self):
      self.__showBtnInfo(self.sender())
      self.ui.textEdit.appendPlainText("移除所有监听的目录和文件\n")

      dirList=self.fileWatcher.directories()
      self.fileWatcher.removePaths(dirList)
      
      fileList=self.fileWatcher.files()
      self.fileWatcher.removePaths(fileList)


   @pyqtSlot()  ##显示监听目录,directories()
   def on_btnWatch_dirs_clicked(self):
      self.__showBtnInfo(self.sender())
      strList=self.fileWatcher.directories()
      self.ui.textEdit.appendPlainText("正在监听的目录:")
      for line in strList:
         self.ui.textEdit.appendPlainText(line)
      self.ui.textEdit.appendPlainText("\n")
      
   @pyqtSlot()  ##显示监听文件,files()
   def on_btnWatch_files_clicked(self):
      self.__showBtnInfo(self.sender())
      strList=self.fileWatcher.files()
      self.ui.textEdit.appendPlainText("正在监听的文件:")
      for line in strList:
         self.ui.textEdit.appendPlainText(line)
      self.ui.textEdit.appendPlainText("\n")
        
##  =============自定义槽函数===============================        

   def do_directoryChanged(self,path):  ##目录发生变化
      self.ui.textEdit.appendPlainText(path)
      self.ui.textEdit.appendPlainText("目录发生了变化\n")
      
   def do_fileChanged(self,path):      ##文件发生变化
      self.ui.textEdit.appendPlainText(path)
      self.ui.textEdit.appendPlainText("文件发生了变化\n")

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   QCoreApplication.setOrganizationName("China University of Petroleum")
   form=QmyMainWindow()             #创建窗体
   form.show()
   sys.exit(app.exec_())
