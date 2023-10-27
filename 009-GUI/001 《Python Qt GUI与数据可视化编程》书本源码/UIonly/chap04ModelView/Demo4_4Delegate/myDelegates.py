from PyQt5.QtWidgets import  QStyledItemDelegate,QDoubleSpinBox,QComboBox

from PyQt5.QtCore import  Qt


## ==============基于QDoubleSpinbox的代理组件====================
class QmyFloatSpinDelegate(QStyledItemDelegate):
   def __init__(self, minV=0,maxV=10000,digi=2,parent=None):
      super().__init__(parent)
      pass
   

## 自定义代理组件必须继承以下4个函数
   def createEditor(self, parent, option, index):
      pass

   def setEditorData(self,editor,index):
      pass
        
   def setModelData(self,editor,model,index):
      pass
    
   def updateEditorGeometry(self,editor,option,index):
      pass


## ==============基于QComboBox的代理组件====================
class QmyComboBoxDelegate(QStyledItemDelegate):
   def __init__(self, parent=None):
      super().__init__(parent)
      pass

   def setItems(self,itemList, isEditable=False):
      pass

## 自定义代理组件必须继承以下4个函数
   def createEditor(self, parent, option, index):
      pass

   def setEditorData(self,editor,index):
      pass
        
   def setModelData(self,editor,model,index):
      pass
    
   def updateEditorGeometry(self,editor,option,index):
      pass
       
        
