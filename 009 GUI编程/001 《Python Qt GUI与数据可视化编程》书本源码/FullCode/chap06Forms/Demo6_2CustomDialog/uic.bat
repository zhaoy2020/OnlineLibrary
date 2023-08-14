echo off

rem 将子目录 QtApp 下的.ui文件复制到当前目录下，并且编译

copy .\QtApp\MainWindow.ui  MainWindow.ui
pyuic5 -o ui_MainWindow.py  MainWindow.ui

copy .\QtApp\QWDialogSize.ui QWDialogSize.ui
pyuic5 -o ui_QWDialogSize.py  QWDialogSize.ui


copy .\QtApp\QWDialogLocate.ui QWDialogLocate.ui
pyuic5 -o ui_QWDialogLocate.py  QWDialogLocate.ui

copy .\QtApp\QWDialogHeaders.ui QWDialogHeaders.ui
pyuic5 -o ui_QWDialogHeaders.py  QWDialogHeaders.ui


rem 编译并复制资源文件

pyrcc5 .\QtApp\res.qrc -o res_rc.py

