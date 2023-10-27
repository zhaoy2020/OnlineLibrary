# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  (pyqtSlot, pyqtSignal, Qt, QFileSystemWatcher,
                        QCoreApplication, QDir, QFileInfo, QFile)

from PyQt5.QtWidgets import  QFileDialog,QMessageBox

##from PyQt5.QtGui import

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass


##  ==============自定义功能函数========================
   def __showBtnInfo(self,btn):  ##显示按钮的text()和toolTip()
      pass
      

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============
   @pyqtSlot()    ##"选择文件"按钮
   def on_btnOpenFile_clicked(self):
      pass

      
   @pyqtSlot()    ##"选择目录"按钮
   def on_btnOpenDir_clicked(self):
      pass


   @pyqtSlot()    ##"清空"按钮
   def on_btnClear_clicked(self):
      pass
      

## =========QFile类 的静态函数===========
   @pyqtSlot()    ##类函数copy()
   def on_btnFile_copy_clicked(self):
      pass
     

   @pyqtSlot()    ##类函数exists()
   def on_btnFile_exists_clicked(self):
      pass

                                          
   @pyqtSlot()    ##类函数remove()
   def on_btnFile_remove_clicked(self):
      pass


   @pyqtSlot()    ##类函数rename()
   def on_btnFile_rename_clicked(self):
      pass
      

## =========QFileInfo类===========
   @pyqtSlot()  ##absoluteFilePath()
   def on_btnInfo_absFilePath_clicked(self):
      pass

   @pyqtSlot()  ##absolutePath()
   def on_btnInfo_absPath_clicked(self):
      pass

   @pyqtSlot()  ##fileName()
   def on_btnInfo_fileName_clicked(self):
      pass

   @pyqtSlot()  ##filePath()
   def on_btnInfo_filePath_clicked(self):
      pass

   @pyqtSlot()  ##size()
   def on_btnInfo_size_clicked(self):
      pass

   @pyqtSlot()  ##path()
   def on_btnInfo_path_clicked(self):
      pass

   @pyqtSlot()  ##baseName()
   def on_btnInfo_baseName_clicked(self):
      pass

   @pyqtSlot()  ##completeBaseName()
   def on_btnInfo_baseName2_clicked(self):
      pass

   @pyqtSlot()  ##suffix()
   def on_btnInfo_suffix_clicked(self):
      pass
      
   @pyqtSlot()  ##completeSuffix()
   def on_btnInfo_suffix2_clicked(self):
      pass

   @pyqtSlot()  ##isDir()
   def on_btnInfo_isDir_clicked(self):
      pass

   @pyqtSlot()  ##isFile()
   def on_btnInfo_isFile_clicked(self):
      pass

   @pyqtSlot()  ##isExecutable()
   def on_btnInfo_isExec_clicked(self):
      pass

   @pyqtSlot()  ##birthTime() ,替代了过时的created()函数
   def on_btnInfo_birthTime_clicked(self):
      pass

   @pyqtSlot()  ##lastModified()
   def on_btnInfo_lastModified_clicked(self):
      pass

   @pyqtSlot()  ##lastRead()
   def on_btnInfo_lastRead_clicked(self):
      pass

   @pyqtSlot()  ##类函数exists()
   def on_btnInfo_exists_clicked(self):
      pass

   @pyqtSlot()  ##接口函数exists()
   def on_btnInfo_exists2_clicked(self):
      pass

## ==================QDir类========================
   @pyqtSlot()  ##tempPath()
   def on_btnDir_tempPath_clicked(self):
      pass

   @pyqtSlot()  ##rootPath()
   def on_btnDir_rootPath_clicked(self):
      pass

   @pyqtSlot()  ##homePath()
   def on_btnDir_homePath_clicked(self):
      pass
         
   @pyqtSlot()  ##drives()
   def on_btnDir_drives_clicked(self):
      pass

   @pyqtSlot()  ##currentPath()
   def on_btnDir_curPath_clicked(self):
      pass

   @pyqtSlot()  ##setCurrent()
   def on_btnDir_setCurPath_clicked(self):
      pass

   @pyqtSlot()  ##mkdir()
   def on_btnDir_mkdir_clicked(self):
      pass

   @pyqtSlot()  ##rmdir()
   def on_btnDir_rmdir_clicked(self):
      pass

   @pyqtSlot()  ##remove()
   def on_btnDir_remove_clicked(self):
      pass

   @pyqtSlot()  ##rename()
   def on_btnDir_rename_clicked(self):
      pass

   @pyqtSlot()  ##setPath(),改换QDir所指的目录
   def on_btnDir_setPath_clicked(self):
      pass

   @pyqtSlot()  ##removeRecursively()
   def on_btnDir_removeALL_clicked(self):
      pass

   @pyqtSlot()  ##absoluteFilePath()
   def on_btnDir_absFilePath_clicked(self):
      pass

   @pyqtSlot()  ##absolutePath()
   def on_btnDir_absPath_clicked(self):
      pass

   @pyqtSlot()  ##canonicalPath()
   def on_btnDir_canonPath_clicked(self):
      pass

   @pyqtSlot()  ##filePath()
   def on_btnDir_filePath_clicked(self):
      pass

   @pyqtSlot()  ##exists()
   def on_btnDir_exists_clicked(self):
      pass

   @pyqtSlot()  ##dirName()
   def on_btnDir_dirName_clicked(self):
      pass

   @pyqtSlot()  ##entryList()dirs
   def on_btnDir_listDir_clicked(self):
      pass

   @pyqtSlot()  ##entryList()files
   def on_btnDir_listFile_clicked(self):
      pass

## ==========QFileSystemWatcher类===================

   @pyqtSlot()  ##addPath()添加监听目录
   def on_btnWatch_addDir_clicked(self):
      pass


   @pyqtSlot()  ##addPaths()添加监听文件
   def on_btnWatch_addFiles_clicked(self):
      pass


   @pyqtSlot()  ##removePaths()移除所有监听的文件和目录
   def on_btnWatch_remove_clicked(self):
      pass


   @pyqtSlot()  ##显示监听目录,directories()
   def on_btnWatch_dirs_clicked(self):
      pass

     
   @pyqtSlot()  ##显示监听文件,files()
   def on_btnWatch_files_clicked(self):
      pass

        
##  =============自定义槽函数===============================        

   def do_directoryChanged(self,path):  ##目录发生变化
      pass

      
   def do_fileChanged(self,path):      ##文件发生变化
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   QCoreApplication.setOrganizationName("China University of Petroleum")
   form=QmyMainWindow()             #创建窗体
   form.show()
   sys.exit(app.exec_())
