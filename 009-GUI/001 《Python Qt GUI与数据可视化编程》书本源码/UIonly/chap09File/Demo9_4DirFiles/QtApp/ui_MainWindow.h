/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QToolBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_10;
    QSplitter *splitter;
    QToolBox *toolBox;
    QWidget *pageFile;
    QVBoxLayout *verticalLayout_8;
    QGroupBox *groupBox_10;
    QVBoxLayout *verticalLayout_11;
    QPushButton *btnFile_exists;
    QPushButton *btnFile_copy;
    QPushButton *btnFile_remove;
    QPushButton *btnFile_rename;
    QWidget *pageFileInfo;
    QVBoxLayout *verticalLayout_5;
    QGroupBox *groupBox_6;
    QGridLayout *gridLayout_4;
    QPushButton *btnInfo_baseName;
    QPushButton *btnInfo_absPath;
    QPushButton *btnInfo_absFilePath;
    QPushButton *btnInfo_baseName2;
    QPushButton *btnInfo_isDir;
    QPushButton *btnInfo_isFile;
    QPushButton *btnInfo_isExec;
    QPushButton *btnInfo_lastModified;
    QPushButton *btnInfo_lastRead;
    QPushButton *btnInfo_fileName;
    QPushButton *btnInfo_exists2;
    QPushButton *btnInfo_suffix;
    QPushButton *btnInfo_exists;
    QPushButton *btnInfo_suffix2;
    QPushButton *btnInfo_filePath;
    QPushButton *btnInfo_path;
    QPushButton *btnInfo_size;
    QPushButton *btnInfo_birthTime;
    QWidget *pageDir;
    QVBoxLayout *verticalLayout_4;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_3;
    QPushButton *btnDir_homePath;
    QPushButton *btnDir_tempPath;
    QPushButton *btnDir_rootPath;
    QPushButton *btnDir_drives;
    QPushButton *btnDir_curPath;
    QPushButton *btnDir_setCurPath;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_2;
    QPushButton *btnDir_mkdir;
    QPushButton *btnDir_remove;
    QPushButton *btnDir_rename;
    QPushButton *btnDir_setPath;
    QPushButton *btnDir_rmdir;
    QPushButton *btnDir_removeALL;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout;
    QPushButton *btnDir_absPath;
    QPushButton *btnDir_canonPath;
    QPushButton *btnDir_filePath;
    QPushButton *btnDir_dirName;
    QPushButton *btnDir_exists;
    QPushButton *btnDir_absFilePath;
    QPushButton *btnDir_listFile;
    QPushButton *btnDir_listDir;
    QWidget *pageWatcher;
    QVBoxLayout *verticalLayout_9;
    QGroupBox *groupBox_11;
    QVBoxLayout *verticalLayout_6;
    QPushButton *btnWatch_addDir;
    QPushButton *btnWatch_addFiles;
    QPushButton *btnWatch_remove;
    QPushButton *btnWatch_dirs;
    QPushButton *btnWatch_files;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *btnOpenFile;
    QPushButton *btnOpenDir;
    QPushButton *btnClear;
    QPushButton *btnClose;
    QGridLayout *gridLayout_6;
    QLabel *label;
    QLineEdit *editFile;
    QLabel *label_2;
    QLineEdit *editDir;
    QLabel *label_3;
    QPlainTextEdit *textEdit;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(896, 530);
        QFont font;
        font.setPointSize(11);
        MainWindow->setFont(font);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_10 = new QVBoxLayout(centralWidget);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        toolBox = new QToolBox(splitter);
        toolBox->setObjectName(QStringLiteral("toolBox"));
        toolBox->setFrameShape(QFrame::StyledPanel);
        pageFile = new QWidget();
        pageFile->setObjectName(QStringLiteral("pageFile"));
        pageFile->setGeometry(QRect(0, 0, 296, 390));
        verticalLayout_8 = new QVBoxLayout(pageFile);
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setContentsMargins(11, 11, 11, 11);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        groupBox_10 = new QGroupBox(pageFile);
        groupBox_10->setObjectName(QStringLiteral("groupBox_10"));
        verticalLayout_11 = new QVBoxLayout(groupBox_10);
        verticalLayout_11->setSpacing(6);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        btnFile_exists = new QPushButton(groupBox_10);
        btnFile_exists->setObjectName(QStringLiteral("btnFile_exists"));
        btnFile_exists->setMinimumSize(QSize(0, 30));

        verticalLayout_11->addWidget(btnFile_exists);

        btnFile_copy = new QPushButton(groupBox_10);
        btnFile_copy->setObjectName(QStringLiteral("btnFile_copy"));
        btnFile_copy->setMinimumSize(QSize(0, 30));

        verticalLayout_11->addWidget(btnFile_copy);

        btnFile_remove = new QPushButton(groupBox_10);
        btnFile_remove->setObjectName(QStringLiteral("btnFile_remove"));
        btnFile_remove->setMinimumSize(QSize(0, 30));

        verticalLayout_11->addWidget(btnFile_remove);

        btnFile_rename = new QPushButton(groupBox_10);
        btnFile_rename->setObjectName(QStringLiteral("btnFile_rename"));
        btnFile_rename->setMinimumSize(QSize(0, 30));

        verticalLayout_11->addWidget(btnFile_rename);

        btnFile_copy->raise();
        btnFile_remove->raise();
        btnFile_rename->raise();
        btnFile_exists->raise();

        verticalLayout_8->addWidget(groupBox_10);

        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/804.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(pageFile, icon, QString::fromUtf8("QFile\347\261\273"));
        pageFileInfo = new QWidget();
        pageFileInfo->setObjectName(QStringLiteral("pageFileInfo"));
        pageFileInfo->setGeometry(QRect(0, 0, 327, 373));
        verticalLayout_5 = new QVBoxLayout(pageFileInfo);
        verticalLayout_5->setSpacing(3);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(3, 3, 3, 3);
        groupBox_6 = new QGroupBox(pageFileInfo);
        groupBox_6->setObjectName(QStringLiteral("groupBox_6"));
        gridLayout_4 = new QGridLayout(groupBox_6);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setHorizontalSpacing(5);
        gridLayout_4->setVerticalSpacing(3);
        gridLayout_4->setContentsMargins(5, 3, 5, 3);
        btnInfo_baseName = new QPushButton(groupBox_6);
        btnInfo_baseName->setObjectName(QStringLiteral("btnInfo_baseName"));
        btnInfo_baseName->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_baseName, 3, 0, 1, 1);

        btnInfo_absPath = new QPushButton(groupBox_6);
        btnInfo_absPath->setObjectName(QStringLiteral("btnInfo_absPath"));
        btnInfo_absPath->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_absPath, 0, 1, 1, 1);

        btnInfo_absFilePath = new QPushButton(groupBox_6);
        btnInfo_absFilePath->setObjectName(QStringLiteral("btnInfo_absFilePath"));
        btnInfo_absFilePath->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_absFilePath, 0, 0, 1, 1);

        btnInfo_baseName2 = new QPushButton(groupBox_6);
        btnInfo_baseName2->setObjectName(QStringLiteral("btnInfo_baseName2"));
        btnInfo_baseName2->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_baseName2, 3, 1, 1, 1);

        btnInfo_isDir = new QPushButton(groupBox_6);
        btnInfo_isDir->setObjectName(QStringLiteral("btnInfo_isDir"));
        btnInfo_isDir->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_isDir, 6, 0, 1, 1);

        btnInfo_isFile = new QPushButton(groupBox_6);
        btnInfo_isFile->setObjectName(QStringLiteral("btnInfo_isFile"));
        btnInfo_isFile->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_isFile, 6, 1, 1, 1);

        btnInfo_isExec = new QPushButton(groupBox_6);
        btnInfo_isExec->setObjectName(QStringLiteral("btnInfo_isExec"));
        btnInfo_isExec->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_isExec, 7, 0, 1, 1);

        btnInfo_lastModified = new QPushButton(groupBox_6);
        btnInfo_lastModified->setObjectName(QStringLiteral("btnInfo_lastModified"));
        btnInfo_lastModified->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_lastModified, 8, 0, 1, 1);

        btnInfo_lastRead = new QPushButton(groupBox_6);
        btnInfo_lastRead->setObjectName(QStringLiteral("btnInfo_lastRead"));
        btnInfo_lastRead->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_lastRead, 8, 1, 1, 1);

        btnInfo_fileName = new QPushButton(groupBox_6);
        btnInfo_fileName->setObjectName(QStringLiteral("btnInfo_fileName"));
        btnInfo_fileName->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_fileName, 1, 0, 1, 1);

        btnInfo_exists2 = new QPushButton(groupBox_6);
        btnInfo_exists2->setObjectName(QStringLiteral("btnInfo_exists2"));
        btnInfo_exists2->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_exists2, 10, 1, 1, 1);

        btnInfo_suffix = new QPushButton(groupBox_6);
        btnInfo_suffix->setObjectName(QStringLiteral("btnInfo_suffix"));
        btnInfo_suffix->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_suffix, 4, 0, 1, 1);

        btnInfo_exists = new QPushButton(groupBox_6);
        btnInfo_exists->setObjectName(QStringLiteral("btnInfo_exists"));
        btnInfo_exists->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_exists, 10, 0, 1, 1);

        btnInfo_suffix2 = new QPushButton(groupBox_6);
        btnInfo_suffix2->setObjectName(QStringLiteral("btnInfo_suffix2"));
        btnInfo_suffix2->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_suffix2, 4, 1, 1, 1);

        btnInfo_filePath = new QPushButton(groupBox_6);
        btnInfo_filePath->setObjectName(QStringLiteral("btnInfo_filePath"));
        btnInfo_filePath->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_filePath, 1, 1, 1, 1);

        btnInfo_path = new QPushButton(groupBox_6);
        btnInfo_path->setObjectName(QStringLiteral("btnInfo_path"));
        btnInfo_path->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_path, 2, 1, 1, 1);

        btnInfo_size = new QPushButton(groupBox_6);
        btnInfo_size->setObjectName(QStringLiteral("btnInfo_size"));
        btnInfo_size->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_size, 2, 0, 1, 1);

        btnInfo_birthTime = new QPushButton(groupBox_6);
        btnInfo_birthTime->setObjectName(QStringLiteral("btnInfo_birthTime"));
        btnInfo_birthTime->setMinimumSize(QSize(0, 30));

        gridLayout_4->addWidget(btnInfo_birthTime, 7, 1, 1, 1);


        verticalLayout_5->addWidget(groupBox_6);

        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/135.JPG"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(pageFileInfo, icon1, QString::fromUtf8("QFileInfo\347\261\273"));
        pageDir = new QWidget();
        pageDir->setObjectName(QStringLiteral("pageDir"));
        pageDir->setGeometry(QRect(0, -104, 324, 477));
        verticalLayout_4 = new QVBoxLayout(pageDir);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        groupBox = new QGroupBox(pageDir);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout_3 = new QGridLayout(groupBox);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        btnDir_homePath = new QPushButton(groupBox);
        btnDir_homePath->setObjectName(QStringLiteral("btnDir_homePath"));
        btnDir_homePath->setMinimumSize(QSize(0, 30));

        gridLayout_3->addWidget(btnDir_homePath, 1, 0, 1, 1);

        btnDir_tempPath = new QPushButton(groupBox);
        btnDir_tempPath->setObjectName(QStringLiteral("btnDir_tempPath"));
        btnDir_tempPath->setMinimumSize(QSize(0, 30));

        gridLayout_3->addWidget(btnDir_tempPath, 0, 0, 1, 1);

        btnDir_rootPath = new QPushButton(groupBox);
        btnDir_rootPath->setObjectName(QStringLiteral("btnDir_rootPath"));
        btnDir_rootPath->setMinimumSize(QSize(0, 30));

        gridLayout_3->addWidget(btnDir_rootPath, 0, 1, 1, 1);

        btnDir_drives = new QPushButton(groupBox);
        btnDir_drives->setObjectName(QStringLiteral("btnDir_drives"));
        btnDir_drives->setMinimumSize(QSize(0, 30));

        gridLayout_3->addWidget(btnDir_drives, 1, 1, 1, 1);

        btnDir_curPath = new QPushButton(groupBox);
        btnDir_curPath->setObjectName(QStringLiteral("btnDir_curPath"));
        btnDir_curPath->setMinimumSize(QSize(0, 30));

        gridLayout_3->addWidget(btnDir_curPath, 2, 0, 1, 1);

        btnDir_setCurPath = new QPushButton(groupBox);
        btnDir_setCurPath->setObjectName(QStringLiteral("btnDir_setCurPath"));
        btnDir_setCurPath->setMinimumSize(QSize(0, 30));

        gridLayout_3->addWidget(btnDir_setCurPath, 2, 1, 1, 1);


        verticalLayout_4->addWidget(groupBox);

        groupBox_4 = new QGroupBox(pageDir);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_2 = new QGridLayout(groupBox_4);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        btnDir_mkdir = new QPushButton(groupBox_4);
        btnDir_mkdir->setObjectName(QStringLiteral("btnDir_mkdir"));
        btnDir_mkdir->setMinimumSize(QSize(0, 30));

        gridLayout_2->addWidget(btnDir_mkdir, 0, 0, 1, 1);

        btnDir_remove = new QPushButton(groupBox_4);
        btnDir_remove->setObjectName(QStringLiteral("btnDir_remove"));
        btnDir_remove->setMinimumSize(QSize(0, 30));

        gridLayout_2->addWidget(btnDir_remove, 2, 0, 1, 1);

        btnDir_rename = new QPushButton(groupBox_4);
        btnDir_rename->setObjectName(QStringLiteral("btnDir_rename"));
        btnDir_rename->setMinimumSize(QSize(0, 30));

        gridLayout_2->addWidget(btnDir_rename, 2, 1, 1, 1);

        btnDir_setPath = new QPushButton(groupBox_4);
        btnDir_setPath->setObjectName(QStringLiteral("btnDir_setPath"));
        btnDir_setPath->setMinimumSize(QSize(0, 30));

        gridLayout_2->addWidget(btnDir_setPath, 3, 0, 1, 1);

        btnDir_rmdir = new QPushButton(groupBox_4);
        btnDir_rmdir->setObjectName(QStringLiteral("btnDir_rmdir"));
        btnDir_rmdir->setMinimumSize(QSize(0, 30));

        gridLayout_2->addWidget(btnDir_rmdir, 0, 1, 1, 1);

        btnDir_removeALL = new QPushButton(groupBox_4);
        btnDir_removeALL->setObjectName(QStringLiteral("btnDir_removeALL"));
        btnDir_removeALL->setMinimumSize(QSize(0, 30));

        gridLayout_2->addWidget(btnDir_removeALL, 3, 1, 1, 1);


        verticalLayout_4->addWidget(groupBox_4);

        groupBox_3 = new QGroupBox(pageDir);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout = new QGridLayout(groupBox_3);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        btnDir_absPath = new QPushButton(groupBox_3);
        btnDir_absPath->setObjectName(QStringLiteral("btnDir_absPath"));
        btnDir_absPath->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_absPath, 0, 1, 1, 1);

        btnDir_canonPath = new QPushButton(groupBox_3);
        btnDir_canonPath->setObjectName(QStringLiteral("btnDir_canonPath"));
        btnDir_canonPath->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_canonPath, 1, 0, 1, 1);

        btnDir_filePath = new QPushButton(groupBox_3);
        btnDir_filePath->setObjectName(QStringLiteral("btnDir_filePath"));
        btnDir_filePath->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_filePath, 1, 1, 1, 1);

        btnDir_dirName = new QPushButton(groupBox_3);
        btnDir_dirName->setObjectName(QStringLiteral("btnDir_dirName"));
        btnDir_dirName->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_dirName, 2, 1, 1, 1);

        btnDir_exists = new QPushButton(groupBox_3);
        btnDir_exists->setObjectName(QStringLiteral("btnDir_exists"));
        btnDir_exists->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_exists, 2, 0, 1, 1);

        btnDir_absFilePath = new QPushButton(groupBox_3);
        btnDir_absFilePath->setObjectName(QStringLiteral("btnDir_absFilePath"));
        btnDir_absFilePath->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_absFilePath, 0, 0, 1, 1);

        btnDir_listFile = new QPushButton(groupBox_3);
        btnDir_listFile->setObjectName(QStringLiteral("btnDir_listFile"));
        btnDir_listFile->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_listFile, 3, 1, 1, 1);

        btnDir_listDir = new QPushButton(groupBox_3);
        btnDir_listDir->setObjectName(QStringLiteral("btnDir_listDir"));
        btnDir_listDir->setMinimumSize(QSize(0, 30));

        gridLayout->addWidget(btnDir_listDir, 3, 0, 1, 1);


        verticalLayout_4->addWidget(groupBox_3);

        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/007.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(pageDir, icon2, QString::fromUtf8("QDir\347\261\273"));
        pageWatcher = new QWidget();
        pageWatcher->setObjectName(QStringLiteral("pageWatcher"));
        pageWatcher->setGeometry(QRect(0, 0, 296, 390));
        verticalLayout_9 = new QVBoxLayout(pageWatcher);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setContentsMargins(11, 11, 11, 11);
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        groupBox_11 = new QGroupBox(pageWatcher);
        groupBox_11->setObjectName(QStringLiteral("groupBox_11"));
        verticalLayout_6 = new QVBoxLayout(groupBox_11);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        btnWatch_addDir = new QPushButton(groupBox_11);
        btnWatch_addDir->setObjectName(QStringLiteral("btnWatch_addDir"));
        btnWatch_addDir->setMinimumSize(QSize(0, 30));

        verticalLayout_6->addWidget(btnWatch_addDir);

        btnWatch_addFiles = new QPushButton(groupBox_11);
        btnWatch_addFiles->setObjectName(QStringLiteral("btnWatch_addFiles"));
        btnWatch_addFiles->setMinimumSize(QSize(0, 30));

        verticalLayout_6->addWidget(btnWatch_addFiles);

        btnWatch_remove = new QPushButton(groupBox_11);
        btnWatch_remove->setObjectName(QStringLiteral("btnWatch_remove"));
        btnWatch_remove->setMinimumSize(QSize(0, 30));

        verticalLayout_6->addWidget(btnWatch_remove);

        btnWatch_dirs = new QPushButton(groupBox_11);
        btnWatch_dirs->setObjectName(QStringLiteral("btnWatch_dirs"));
        btnWatch_dirs->setMinimumSize(QSize(0, 30));

        verticalLayout_6->addWidget(btnWatch_dirs);

        btnWatch_files = new QPushButton(groupBox_11);
        btnWatch_files->setObjectName(QStringLiteral("btnWatch_files"));
        btnWatch_files->setMinimumSize(QSize(0, 30));

        verticalLayout_6->addWidget(btnWatch_files);


        verticalLayout_9->addWidget(groupBox_11);

        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/714.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(pageWatcher, icon3, QString::fromUtf8("QFileSystemWatcher\347\261\273"));
        splitter->addWidget(toolBox);
        groupBox_2 = new QGroupBox(splitter);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setFlat(false);
        verticalLayout = new QVBoxLayout(groupBox_2);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        btnOpenFile = new QPushButton(groupBox_2);
        btnOpenFile->setObjectName(QStringLiteral("btnOpenFile"));
        btnOpenFile->setMinimumSize(QSize(0, 30));
        btnOpenFile->setIcon(icon2);

        horizontalLayout->addWidget(btnOpenFile);

        btnOpenDir = new QPushButton(groupBox_2);
        btnOpenDir->setObjectName(QStringLiteral("btnOpenDir"));
        btnOpenDir->setMinimumSize(QSize(0, 30));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/122.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnOpenDir->setIcon(icon4);

        horizontalLayout->addWidget(btnOpenDir);

        btnClear = new QPushButton(groupBox_2);
        btnClear->setObjectName(QStringLiteral("btnClear"));
        btnClear->setMinimumSize(QSize(0, 30));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/212.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnClear->setIcon(icon5);

        horizontalLayout->addWidget(btnClear);

        btnClose = new QPushButton(groupBox_2);
        btnClose->setObjectName(QStringLiteral("btnClose"));
        btnClose->setMinimumSize(QSize(0, 30));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnClose->setIcon(icon6);

        horizontalLayout->addWidget(btnClose);


        verticalLayout->addLayout(horizontalLayout);

        gridLayout_6 = new QGridLayout();
        gridLayout_6->setSpacing(6);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_6->addWidget(label, 0, 0, 1, 1);

        editFile = new QLineEdit(groupBox_2);
        editFile->setObjectName(QStringLiteral("editFile"));
        editFile->setClearButtonEnabled(true);

        gridLayout_6->addWidget(editFile, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_6->addWidget(label_2, 1, 0, 1, 1);

        editDir = new QLineEdit(groupBox_2);
        editDir->setObjectName(QStringLiteral("editDir"));
        editDir->setClearButtonEnabled(true);

        gridLayout_6->addWidget(editDir, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_6->addWidget(label_3, 2, 0, 1, 1);

        textEdit = new QPlainTextEdit(groupBox_2);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        QFont font1;
        font1.setPointSize(12);
        textEdit->setFont(font1);
        textEdit->setLineWrapMode(QPlainTextEdit::WidgetWidth);

        gridLayout_6->addWidget(textEdit, 2, 1, 1, 1);


        verticalLayout->addLayout(gridLayout_6);

        splitter->addWidget(groupBox_2);

        verticalLayout_10->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);
        QObject::connect(btnClose, SIGNAL(clicked()), MainWindow, SLOT(close()));

        toolBox->setCurrentIndex(3);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo9_4\357\274\214\347\233\256\345\275\225\344\270\216\346\226\207\344\273\266\346\223\215\344\275\234", nullptr));
        groupBox_10->setTitle(QApplication::translate("MainWindow", "\351\235\231\346\200\201\345\207\275\346\225\260", nullptr));
#ifndef QT_NO_TOOLTIP
        btnFile_exists->setToolTip(QApplication::translate("MainWindow", "Returns true if the file specified by fileName exists; otherwise returns false.", nullptr));
#endif // QT_NO_TOOLTIP
        btnFile_exists->setText(QApplication::translate("MainWindow", "exists()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnFile_copy->setToolTip(QApplication::translate("MainWindow", "Copies the file fileName to newName.", nullptr));
#endif // QT_NO_TOOLTIP
        btnFile_copy->setText(QApplication::translate("MainWindow", "copy()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnFile_remove->setToolTip(QApplication::translate("MainWindow", "Removes the file specified by the fileName given.", nullptr));
#endif // QT_NO_TOOLTIP
        btnFile_remove->setText(QApplication::translate("MainWindow", "remove()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnFile_rename->setToolTip(QApplication::translate("MainWindow", "Renames the file oldName to newName. ", nullptr));
#endif // QT_NO_TOOLTIP
        btnFile_rename->setText(QApplication::translate("MainWindow", "rename()", nullptr));
        toolBox->setItemText(toolBox->indexOf(pageFile), QApplication::translate("MainWindow", "QFile\347\261\273", nullptr));
        groupBox_6->setTitle(QApplication::translate("MainWindow", "\346\226\207\344\273\266\344\277\241\346\201\257", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_baseName->setToolTip(QApplication::translate("MainWindow", "The base name consists of all characters in the file up to (but not including) the first '.' character", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_baseName->setText(QApplication::translate("MainWindow", "baseName()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_absPath->setToolTip(QApplication::translate("MainWindow", "Returns a file's path absolute path. This doesn't include the file name", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_absPath->setText(QApplication::translate("MainWindow", "absolutePath() ", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_absFilePath->setToolTip(QApplication::translate("MainWindow", "Returns an absolute path including the file name", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_absFilePath->setText(QApplication::translate("MainWindow", "absoluteFilePath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_baseName2->setToolTip(QApplication::translate("MainWindow", "The complete base name consists of all characters in the file up to (but not including) the last '.' character", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_baseName2->setText(QApplication::translate("MainWindow", "completeBaseName()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_isDir->setToolTip(QApplication::translate("MainWindow", "Returns true if this object points to a directory or to a symbolic link to a directory; otherwise returns false", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_isDir->setText(QApplication::translate("MainWindow", "isDir()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_isFile->setToolTip(QApplication::translate("MainWindow", "Returns true if this object points to a file or to a symbolic link to a file", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_isFile->setText(QApplication::translate("MainWindow", "isFile()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_isExec->setToolTip(QApplication::translate("MainWindow", "Returns true if the file is executable; otherwise returns false.", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_isExec->setText(QApplication::translate("MainWindow", "isExecutable()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_lastModified->setToolTip(QApplication::translate("MainWindow", "Returns the date and time when the file was last modified.", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_lastModified->setText(QApplication::translate("MainWindow", "lastModified()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_lastRead->setToolTip(QApplication::translate("MainWindow", "Returns the date and time when the file was last read (accessed).", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_lastRead->setText(QApplication::translate("MainWindow", "lastRead()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_fileName->setToolTip(QApplication::translate("MainWindow", "Returns the name of the file, excluding the path.", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_fileName->setText(QApplication::translate("MainWindow", "fileName()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_exists2->setToolTip(QApplication::translate("MainWindow", "Returns true if the file exists; otherwise returns false", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_exists2->setText(QApplication::translate("MainWindow", "exists()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_suffix->setToolTip(QApplication::translate("MainWindow", "The suffix consists of all characters in the file after (but not including) the last '.'", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_suffix->setText(QApplication::translate("MainWindow", "suffix()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_exists->setToolTip(QApplication::translate("MainWindow", "Returns true if the file exists; otherwise returns false", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_exists->setText(QApplication::translate("MainWindow", "\351\235\231\346\200\201\345\207\275\346\225\260exists()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_suffix2->setToolTip(QApplication::translate("MainWindow", "The complete suffix consists of all characters in the file after (but not including) the first '.'", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_suffix2->setText(QApplication::translate("MainWindow", "completeSuffix()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_filePath->setToolTip(QApplication::translate("MainWindow", "Returns the file name, including the path (which may be absolute or relative).", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_filePath->setText(QApplication::translate("MainWindow", "filePath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_path->setToolTip(QApplication::translate("MainWindow", "Returns the file's path. This doesn't include the file name.", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_path->setText(QApplication::translate("MainWindow", "path()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_size->setToolTip(QApplication::translate("MainWindow", "Returns the file size in bytes.", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_size->setText(QApplication::translate("MainWindow", "size()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnInfo_birthTime->setToolTip(QApplication::translate("MainWindow", "Returns the date and time when the file was created / born", nullptr));
#endif // QT_NO_TOOLTIP
        btnInfo_birthTime->setText(QApplication::translate("MainWindow", "birthTime()", nullptr));
        toolBox->setItemText(toolBox->indexOf(pageFileInfo), QApplication::translate("MainWindow", "QFileInfo\347\261\273", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "\351\235\231\346\200\201\345\207\275\346\225\260", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_homePath->setToolTip(QApplication::translate("MainWindow", "Returns the absolute path of the user's home directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_homePath->setText(QApplication::translate("MainWindow", "homePath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_tempPath->setToolTip(QApplication::translate("MainWindow", "Returns the absolute path of the system's temporary directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_tempPath->setText(QApplication::translate("MainWindow", "tempPath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_rootPath->setToolTip(QApplication::translate("MainWindow", "Returns the absolute path of the root directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_rootPath->setText(QApplication::translate("MainWindow", "rootPath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_drives->setToolTip(QApplication::translate("MainWindow", "Returns a list of the root directories on this system", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_drives->setText(QApplication::translate("MainWindow", "drives()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_curPath->setToolTip(QApplication::translate("MainWindow", "Returns the absolute path of the application's current directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_curPath->setText(QApplication::translate("MainWindow", "currentPath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_setCurPath->setToolTip(QApplication::translate("MainWindow", "Sets the application's current working directory to path", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_setCurPath->setText(QApplication::translate("MainWindow", "setCurrent()", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\346\226\207\344\273\266\346\210\226\347\233\256\345\275\225\346\223\215\344\275\234", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_mkdir->setToolTip(QApplication::translate("MainWindow", "Creates a sub-directory called dirName.", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_mkdir->setText(QApplication::translate("MainWindow", "mkdir()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_remove->setToolTip(QApplication::translate("MainWindow", "Removes the file, fileName.", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_remove->setText(QApplication::translate("MainWindow", "remove()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_rename->setToolTip(QApplication::translate("MainWindow", "Renames a file or directory from oldName to newName", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_rename->setText(QApplication::translate("MainWindow", "rename()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_setPath->setToolTip(QApplication::translate("MainWindow", "Sets the path of the directory to path", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_setPath->setText(QApplication::translate("MainWindow", "setPath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_rmdir->setToolTip(QApplication::translate("MainWindow", "Removes the directory specified by dirName.", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_rmdir->setText(QApplication::translate("MainWindow", "rmdir()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_removeALL->setToolTip(QApplication::translate("MainWindow", "Removes the directory, including all its contents.", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_removeALL->setText(QApplication::translate("MainWindow", "removeRecursively()", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\346\226\207\344\273\266\346\210\226\347\233\256\345\275\225\344\277\241\346\201\257", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_absPath->setToolTip(QApplication::translate("MainWindow", "Returns the absolute path (a path that starts with \"/\" or with a drive specification)", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_absPath->setText(QApplication::translate("MainWindow", "absolutePath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_canonPath->setToolTip(QApplication::translate("MainWindow", "Returns the canonical path, i.e. a path without symbolic links or redundant \".\" or \"..\" elements.", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_canonPath->setText(QApplication::translate("MainWindow", "canonicalPath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_filePath->setToolTip(QApplication::translate("MainWindow", "Returns the path name of a file in the directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_filePath->setText(QApplication::translate("MainWindow", "filePath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_dirName->setToolTip(QApplication::translate("MainWindow", "Returns the name of the directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_dirName->setText(QApplication::translate("MainWindow", "dirName()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_exists->setToolTip(QApplication::translate("MainWindow", "Returns true if the directory exists", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_exists->setText(QApplication::translate("MainWindow", "exists()\347\233\256\345\275\225", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_absFilePath->setToolTip(QApplication::translate("MainWindow", "Returns the absolute path name of a file in the directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_absFilePath->setText(QApplication::translate("MainWindow", "absoluteFilePath()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_listFile->setToolTip(QApplication::translate("MainWindow", "Returns a list of the names of all the files and directories in the directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_listFile->setText(QApplication::translate("MainWindow", "entryList(file)", nullptr));
#ifndef QT_NO_TOOLTIP
        btnDir_listDir->setToolTip(QApplication::translate("MainWindow", "Returns a list of the names of all the files and directories in the directory", nullptr));
#endif // QT_NO_TOOLTIP
        btnDir_listDir->setText(QApplication::translate("MainWindow", "entryList(dir)", nullptr));
        toolBox->setItemText(toolBox->indexOf(pageDir), QApplication::translate("MainWindow", "QDir\347\261\273", nullptr));
        groupBox_11->setTitle(QString());
#ifndef QT_NO_ACCESSIBILITY
        btnWatch_addDir->setAccessibleDescription(QApplication::translate("MainWindow", "Adds path to the file system watcher if path exists. ", nullptr));
#endif // QT_NO_ACCESSIBILITY
        btnWatch_addDir->setText(QApplication::translate("MainWindow", "addPath()\346\267\273\345\212\240\347\233\221\345\220\254\347\233\256\345\275\225", nullptr));
#ifndef QT_NO_TOOLTIP
        btnWatch_addFiles->setToolTip(QApplication::translate("MainWindow", "Adds each path in paths to the file system watcher. ", nullptr));
#endif // QT_NO_TOOLTIP
        btnWatch_addFiles->setText(QApplication::translate("MainWindow", "addPaths()\346\267\273\345\212\240\347\233\221\345\220\254\346\226\207\344\273\266", nullptr));
#ifndef QT_NO_TOOLTIP
        btnWatch_remove->setToolTip(QApplication::translate("MainWindow", "Removes the specified path from the file system watcher.", nullptr));
#endif // QT_NO_TOOLTIP
        btnWatch_remove->setText(QApplication::translate("MainWindow", "removePaths()\347\247\273\351\231\244\346\211\200\346\234\211\347\233\221\345\220\254\351\241\271", nullptr));
#ifndef QT_NO_TOOLTIP
        btnWatch_dirs->setToolTip(QApplication::translate("MainWindow", "Returns a list of paths to directories that are being watched.", nullptr));
#endif // QT_NO_TOOLTIP
        btnWatch_dirs->setText(QApplication::translate("MainWindow", "directories()", nullptr));
#ifndef QT_NO_TOOLTIP
        btnWatch_files->setToolTip(QApplication::translate("MainWindow", "Returns a list of paths to files that are being watched.", nullptr));
#endif // QT_NO_TOOLTIP
        btnWatch_files->setText(QApplication::translate("MainWindow", "files()", nullptr));
        toolBox->setItemText(toolBox->indexOf(pageWatcher), QApplication::translate("MainWindow", "QFileSystemWatcher\347\261\273", nullptr));
        groupBox_2->setTitle(QString());
        btnOpenFile->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\346\226\207\344\273\266", nullptr));
        btnOpenDir->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\347\233\256\345\275\225", nullptr));
#ifndef QT_NO_TOOLTIP
        btnClear->setToolTip(QApplication::translate("MainWindow", "\346\270\205\347\251\272\344\277\241\346\201\257\350\276\223\345\207\272\346\226\207\346\234\254\346\241\206", nullptr));
#endif // QT_NO_TOOLTIP
        btnClear->setText(QApplication::translate("MainWindow", "\346\270\205\347\251\272", nullptr));
        btnClose->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
        label->setText(QApplication::translate("MainWindow", "\346\226\207 \344\273\266 ", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\347\233\256 \345\275\225 ", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\344\277\241 \346\201\257 ", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
