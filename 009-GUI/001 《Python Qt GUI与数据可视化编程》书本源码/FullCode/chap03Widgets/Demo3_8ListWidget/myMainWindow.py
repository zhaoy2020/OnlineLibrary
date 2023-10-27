import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
                       QListWidgetItem, QMenu, QToolButton)

from PyQt5.QtGui import  QIcon, QCursor

from PyQt5.QtCore import  pyqtSlot, Qt

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.setCentralWidget(self.ui.splitter)

      self.__setActionsForButton()
      self.__createSelectionPopMenu()

      self.__FlagEditable =(Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                            | Qt.ItemIsEnabled | Qt.ItemIsEditable)
      self.__FlagNotEditable =( Qt.ItemIsSelectable
                           | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        

##  ================自定义功能函数============================
   def __setActionsForButton(self):  ##为界面上的ToolButton按钮设置Actions
      self.ui.btnList_Ini.setDefaultAction(self.ui.actList_Ini)
      self.ui.btnList_Clear.setDefaultAction(self.ui.actList_Clear)

      self.ui.btnList_Insert.setDefaultAction(self.ui.actList_Insert)
      self.ui.btnList_Append.setDefaultAction(self.ui.actList_Append)
      self.ui.btnList_Delete.setDefaultAction(self.ui.actList_Delete)

      self.ui.btnSel_ALL.setDefaultAction(self.ui.actSel_ALL)
      self.ui.btnSel_None.setDefaultAction(self.ui.actSel_None)
      self.ui.btnSel_Invs.setDefaultAction(self.ui.actSel_Invs)

   def __createSelectionPopMenu(self):  ##创建ToolButton的下拉菜单
      menuSelection=QMenu(self)     #下拉菜单
      menuSelection.addAction(self.ui.actSel_ALL)
      menuSelection.addAction(self.ui.actSel_None)
      menuSelection.addAction(self.ui.actSel_Invs)

      #listWidget上方的 btnSelectItem 按钮
      self.ui.btnSelectItem.setPopupMode(QToolButton.MenuButtonPopup)
      ##        self.ui.btnSelectItem.setPopupMode(QToolButton.InstantPopup)
      self.ui.btnSelectItem.setToolButtonStyle(
                  Qt.ToolButtonTextBesideIcon)
      self.ui.btnSelectItem.setDefaultAction(self.ui.actSelPopMenu)
      self.ui.btnSelectItem.setMenu(menuSelection) #设置下拉菜单

      ##工具栏上的下拉式菜单按钮
      toolBtn=QToolButton(self)
      toolBtn.setPopupMode(QToolButton.InstantPopup)
      toolBtn.setDefaultAction(self.ui.actSelPopMenu)
      toolBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
      toolBtn.setMenu(menuSelection) #设置下拉菜单
      self.ui.mainToolBar.addWidget(toolBtn)
        
      #工具栏添加分隔条，和“退出”按钮
      self.ui.mainToolBar.addSeparator()
      self.ui.mainToolBar.addAction(self.ui.actQuit)

        
##  =========connectSlotsByName() 自动连接的槽函数===========        
    
   @pyqtSlot()  ##初始化列表
   def on_actList_Ini_triggered(self):    
      icon = QIcon(":/icons/images/724.bmp")
      editable=self.ui.chkBoxList_Editable.isChecked()

      if (editable == True):
         Flag=self.__FlagEditable    #可编辑
      else:
         Flag=self.__FlagNotEditable #不可编辑

      self.ui.listWidget.clear()     #清除列表
      for i in range(10):
         itemStr="Item %d"%i
         aItem=QListWidgetItem()
         aItem.setText(itemStr)
         aItem.setIcon(icon)
         aItem.setCheckState(Qt.Checked)
         aItem.setFlags(Flag)        #项的flags
         self.ui.listWidget.addItem(aItem)


   @pyqtSlot()    ##插入一项
   def on_actList_Insert_triggered(self):  
      icon = QIcon(":/icons/images/724.bmp")
      editable=self.ui.chkBoxList_Editable.isChecked()

      if (editable == True):
         Flag=self.__FlagEditable    #可编辑
      else:
         Flag=self.__FlagNotEditable #不可编辑

      aItem=QListWidgetItem()
      aItem.setText("Inserted Item")
      aItem.setIcon(icon)
      aItem.setCheckState(Qt.Checked)
      aItem.setFlags(Flag)            #项的flags
      curRow=self.ui.listWidget.currentRow()  #当前行
      self.ui.listWidget.insertItem(curRow,aItem)

        
   @pyqtSlot()  ##添加一项
   def on_actList_Append_triggered(self):    
      icon = QIcon(":/icons/images/724.bmp")
      editable=self.ui.chkBoxList_Editable.isChecked()

      if (editable == True):
         Flag=self.__FlagEditable 
      else:
         Flag=self.__FlagNotEditable 

      aItem=QListWidgetItem()
      aItem.setText("Appended Item")
      aItem.setIcon(icon)
      aItem.setCheckState(Qt.Checked)
      aItem.setFlags(Flag)
      self.ui.listWidget.addItem(aItem)
            

   @pyqtSlot()   ##删除当前项
   def on_actList_Delete_triggered(self):  
      row=self.ui.listWidget.currentRow()
      self.ui.listWidget.takeItem(row)  #剔除当前项，Python自动删除
        
   @pyqtSlot()   ##清空列表
   def on_actList_Clear_triggered(self):   
      self.ui.listWidget.clear()

   @pyqtSlot()
   def on_actSel_ALL_triggered(self):     ##全选
      for i in range(self.ui.listWidget.count()):
         aItem=self.ui.listWidget.item(i)
         aItem.setCheckState(Qt.Checked)
            
   @pyqtSlot()
   def on_actSel_None_triggered(self):     ##全不选
      for i in range(self.ui.listWidget.count()):
         aItem=self.ui.listWidget.item(i)
         aItem.setCheckState(Qt.Unchecked)
    
   @pyqtSlot()
   def on_actSel_Invs_triggered(self):     ##反选
      for i in range(self.ui.listWidget.count()):
         aItem=self.ui.listWidget.item(i)
         if (aItem.checkState() != Qt.Checked):
            aItem.setCheckState(Qt.Checked)
         else:
            aItem.setCheckState(Qt.Unchecked)

   @pyqtSlot(bool)
   def on_chkBoxList_Editable_clicked(self,checked):   ##改变可编辑状态
      if (checked == True):
         Flag=self.__FlagEditable
      else:
         Flag=self.__FlagNotEditable
         
      for i in range(self.ui.listWidget.count()):
         aItem=self.ui.listWidget.item(i)
         aItem.setFlags(Flag)

   def on_listWidget_currentItemChanged(self,current,previous):    ##当前项切换变化
      strInfo=""
      if (current!=None):
         if (previous==None):
            strInfo="当前:"+current.text()
         else:
            strInfo="前一项:"+previous.text()+"；当前项："+current.text()
      self.ui.editCurItemText.setText(strInfo)

   def on_listWidget_customContextMenuRequested(self,pos):  ##右键快捷菜单
      menuList=QMenu(self)    #创建菜单

      menuList.addAction(self.ui.actList_Ini)
      menuList.addAction(self.ui.actList_Clear)

      menuList.addAction(self.ui.actList_Insert)
      menuList.addAction(self.ui.actList_Append)
      menuList.addAction(self.ui.actList_Delete)
      menuList.addSeparator()

      menuList.addAction(self.ui.actSel_ALL)
      menuList.addAction(self.ui.actSel_None)
      menuList.addAction(self.ui.actSel_Invs)

      menuList.exec(QCursor.pos())  #显示菜单
        
##  ================自定义槽函数===============================        

   
##  ================窗体测试程序 ============================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
