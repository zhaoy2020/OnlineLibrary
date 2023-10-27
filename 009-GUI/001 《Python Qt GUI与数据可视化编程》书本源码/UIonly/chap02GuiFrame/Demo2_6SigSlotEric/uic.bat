echo off

rem 将子目录 QtApp 下的.ui文件复制到当前目录下
copy .\QtApp\Dialog.ui  Dialog.ui

rem 用pyuic5编译.ui文件
pyuic5 -o ui_Dialog.py  Dialog.ui