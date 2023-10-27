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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actClearOutput;
    QAction *actClose;
    QAction *actSaveALL;
    QAction *actReadALL;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QToolButton *btnFile;
    QLineEdit *editFilename;
    QHBoxLayout *horizontalLayout_4;
    QGroupBox *groupBox_3;
    QHBoxLayout *horizontalLayout;
    QRadioButton *radio_LittleEndian;
    QRadioButton *radio_BigEndian;
    QGroupBox *groupBox_4;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *radio_Single;
    QRadioButton *radio_Double;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLineEdit *edit_Double;
    QLabel *label_13;
    QLineEdit *editQStr_In;
    QPushButton *btnQStr_Write;
    QPushButton *btnQStr_Read;
    QLineEdit *editQStr_Out;
    QLabel *label_10;
    QCheckBox *chkBox_In;
    QPushButton *btnBool_Write;
    QPushButton *btnBool_Read;
    QSpinBox *spin_UInt8;
    QSpinBox *spin_Int8;
    QPushButton *btnInt8_Write;
    QLineEdit *edit_Int8;
    QLabel *label;
    QPushButton *btnStr_Read;
    QLabel *label_11;
    QDoubleSpinBox *spin_Float;
    QPushButton *btnFloat_Write;
    QPushButton *btnFloat_Read;
    QLineEdit *edit_Float;
    QLabel *label_12;
    QDoubleSpinBox *spin_Double;
    QLabel *label_2;
    QLineEdit *edit_UInt8;
    QLabel *label_4;
    QSpinBox *spin_Int16;
    QPushButton *btnFont_Read;
    QLineEdit *editFont_Out;
    QLabel *label_16;
    QPushButton *btnColor_In;
    QPushButton *btnColor_Write;
    QPushButton *btnColor_Read;
    QLineEdit *editColor_Out;
    QLabel *label_6;
    QSpinBox *spin_Int32;
    QPushButton *btnInt32_Write;
    QPushButton *btnInt32_Read;
    QLineEdit *edit_Int32;
    QLabel *label_7;
    QSpinBox *spin_Int64;
    QPushButton *btnInt64_Write;
    QPushButton *btnInt8_Read;
    QLabel *label_15;
    QPushButton *btnFont_In;
    QPushButton *btnFont_Write;
    QPushButton *btnDouble_Write;
    QPushButton *btnDouble_Read;
    QLineEdit *editStr_Out;
    QLineEdit *editStr_In;
    QPushButton *btnInt64_Read;
    QLineEdit *edit_Int64;
    QLabel *label_8;
    QSpinBox *spin_Int;
    QPushButton *btnInt_Write;
    QPushButton *btnInt_Read;
    QLineEdit *edit_Int;
    QPushButton *btnUInt8_Write;
    QPushButton *btnUInt8_Read;
    QPushButton *btnInt16_Write;
    QPushButton *btnInt16_Read;
    QLineEdit *edit_Int16;
    QLabel *label_5;
    QSpinBox *spin_UInt16;
    QPushButton *btnUInt16_Write;
    QPushButton *btnUIn16_Read;
    QLineEdit *edit_UInt16;
    QCheckBox *chkBox_Out;
    QLabel *label_3;
    QPushButton *btnStr_Write;
    QToolBar *toolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(614, 567);
        actClearOutput = new QAction(MainWindow);
        actClearOutput->setObjectName(QStringLiteral("actClearOutput"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/214.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClearOutput->setIcon(icon);
        actClose = new QAction(MainWindow);
        actClose->setObjectName(QStringLiteral("actClose"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon1);
        actSaveALL = new QAction(MainWindow);
        actSaveALL->setObjectName(QStringLiteral("actSaveALL"));
        actSaveALL->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/saveall1.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSaveALL->setIcon(icon2);
        actReadALL = new QAction(MainWindow);
        actReadALL->setObjectName(QStringLiteral("actReadALL"));
        actReadALL->setEnabled(false);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/upfold1.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actReadALL->setIcon(icon3);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setMaximumSize(QSize(16777215, 90));
        verticalLayout_2 = new QVBoxLayout(groupBox_2);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(-1, 5, -1, 5);
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        btnFile = new QToolButton(groupBox_2);
        btnFile->setObjectName(QStringLiteral("btnFile"));
        btnFile->setMinimumSize(QSize(0, 22));

        horizontalLayout_3->addWidget(btnFile);

        editFilename = new QLineEdit(groupBox_2);
        editFilename->setObjectName(QStringLiteral("editFilename"));

        horizontalLayout_3->addWidget(editFilename);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        groupBox_3 = new QGroupBox(groupBox_2);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        horizontalLayout = new QHBoxLayout(groupBox_3);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 5, -1, 5);
        radio_LittleEndian = new QRadioButton(groupBox_3);
        radio_LittleEndian->setObjectName(QStringLiteral("radio_LittleEndian"));
        radio_LittleEndian->setChecked(true);

        horizontalLayout->addWidget(radio_LittleEndian);

        radio_BigEndian = new QRadioButton(groupBox_3);
        radio_BigEndian->setObjectName(QStringLiteral("radio_BigEndian"));

        horizontalLayout->addWidget(radio_BigEndian);


        horizontalLayout_4->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(groupBox_2);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        horizontalLayout_2 = new QHBoxLayout(groupBox_4);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, 5, -1, 5);
        radio_Single = new QRadioButton(groupBox_4);
        radio_Single->setObjectName(QStringLiteral("radio_Single"));
        radio_Single->setChecked(true);

        horizontalLayout_2->addWidget(radio_Single);

        radio_Double = new QRadioButton(groupBox_4);
        radio_Double->setObjectName(QStringLiteral("radio_Double"));

        horizontalLayout_2->addWidget(radio_Double);


        horizontalLayout_4->addWidget(groupBox_4);


        verticalLayout_2->addLayout(horizontalLayout_4);


        verticalLayout->addWidget(groupBox_2);

        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setHorizontalSpacing(6);
        gridLayout->setVerticalSpacing(5);
        edit_Double = new QLineEdit(groupBox);
        edit_Double->setObjectName(QStringLiteral("edit_Double"));
        edit_Double->setReadOnly(true);

        gridLayout->addWidget(edit_Double, 9, 4, 1, 1);

        label_13 = new QLabel(groupBox);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_13, 10, 0, 1, 1);

        editQStr_In = new QLineEdit(groupBox);
        editQStr_In->setObjectName(QStringLiteral("editQStr_In"));

        gridLayout->addWidget(editQStr_In, 10, 1, 1, 1);

        btnQStr_Write = new QPushButton(groupBox);
        btnQStr_Write->setObjectName(QStringLiteral("btnQStr_Write"));

        gridLayout->addWidget(btnQStr_Write, 10, 2, 1, 1);

        btnQStr_Read = new QPushButton(groupBox);
        btnQStr_Read->setObjectName(QStringLiteral("btnQStr_Read"));

        gridLayout->addWidget(btnQStr_Read, 10, 3, 1, 1);

        editQStr_Out = new QLineEdit(groupBox);
        editQStr_Out->setObjectName(QStringLiteral("editQStr_Out"));
        editQStr_Out->setReadOnly(true);

        gridLayout->addWidget(editQStr_Out, 10, 4, 1, 1);

        label_10 = new QLabel(groupBox);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_10, 7, 0, 1, 1);

        chkBox_In = new QCheckBox(groupBox);
        chkBox_In->setObjectName(QStringLiteral("chkBox_In"));

        gridLayout->addWidget(chkBox_In, 7, 1, 1, 1);

        btnBool_Write = new QPushButton(groupBox);
        btnBool_Write->setObjectName(QStringLiteral("btnBool_Write"));

        gridLayout->addWidget(btnBool_Write, 7, 2, 1, 1);

        btnBool_Read = new QPushButton(groupBox);
        btnBool_Read->setObjectName(QStringLiteral("btnBool_Read"));

        gridLayout->addWidget(btnBool_Read, 7, 3, 1, 1);

        spin_UInt8 = new QSpinBox(groupBox);
        spin_UInt8->setObjectName(QStringLiteral("spin_UInt8"));
        spin_UInt8->setMinimum(0);
        spin_UInt8->setMaximum(255);
        spin_UInt8->setValue(220);

        gridLayout->addWidget(spin_UInt8, 1, 1, 1, 1);

        spin_Int8 = new QSpinBox(groupBox);
        spin_Int8->setObjectName(QStringLiteral("spin_Int8"));
        spin_Int8->setMinimum(-128);
        spin_Int8->setMaximum(127);
        spin_Int8->setValue(56);

        gridLayout->addWidget(spin_Int8, 0, 1, 1, 1);

        btnInt8_Write = new QPushButton(groupBox);
        btnInt8_Write->setObjectName(QStringLiteral("btnInt8_Write"));

        gridLayout->addWidget(btnInt8_Write, 0, 2, 1, 1);

        edit_Int8 = new QLineEdit(groupBox);
        edit_Int8->setObjectName(QStringLiteral("edit_Int8"));
        edit_Int8->setReadOnly(true);

        gridLayout->addWidget(edit_Int8, 0, 4, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label, 0, 0, 1, 1);

        btnStr_Read = new QPushButton(groupBox);
        btnStr_Read->setObjectName(QStringLiteral("btnStr_Read"));

        gridLayout->addWidget(btnStr_Read, 11, 3, 1, 1);

        label_11 = new QLabel(groupBox);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_11, 8, 0, 1, 1);

        spin_Float = new QDoubleSpinBox(groupBox);
        spin_Float->setObjectName(QStringLiteral("spin_Float"));
        spin_Float->setDecimals(4);
        spin_Float->setMinimum(-600000);
        spin_Float->setMaximum(6e+6);
        spin_Float->setValue(15.26);

        gridLayout->addWidget(spin_Float, 8, 1, 1, 1);

        btnFloat_Write = new QPushButton(groupBox);
        btnFloat_Write->setObjectName(QStringLiteral("btnFloat_Write"));

        gridLayout->addWidget(btnFloat_Write, 8, 2, 1, 1);

        btnFloat_Read = new QPushButton(groupBox);
        btnFloat_Read->setObjectName(QStringLiteral("btnFloat_Read"));

        gridLayout->addWidget(btnFloat_Read, 8, 3, 1, 1);

        edit_Float = new QLineEdit(groupBox);
        edit_Float->setObjectName(QStringLiteral("edit_Float"));
        edit_Float->setReadOnly(true);

        gridLayout->addWidget(edit_Float, 8, 4, 1, 1);

        label_12 = new QLabel(groupBox);
        label_12->setObjectName(QStringLiteral("label_12"));
        label_12->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_12, 9, 0, 1, 1);

        spin_Double = new QDoubleSpinBox(groupBox);
        spin_Double->setObjectName(QStringLiteral("spin_Double"));
        spin_Double->setDecimals(4);
        spin_Double->setMinimum(-6e+6);
        spin_Double->setMaximum(6e+6);
        spin_Double->setValue(-56.253);

        gridLayout->addWidget(spin_Double, 9, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        edit_UInt8 = new QLineEdit(groupBox);
        edit_UInt8->setObjectName(QStringLiteral("edit_UInt8"));
        edit_UInt8->setReadOnly(true);

        gridLayout->addWidget(edit_UInt8, 1, 4, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_4, 2, 0, 1, 1);

        spin_Int16 = new QSpinBox(groupBox);
        spin_Int16->setObjectName(QStringLiteral("spin_Int16"));
        spin_Int16->setMinimum(-32767);
        spin_Int16->setMaximum(32767);
        spin_Int16->setValue(-1522);

        gridLayout->addWidget(spin_Int16, 2, 1, 1, 1);

        btnFont_Read = new QPushButton(groupBox);
        btnFont_Read->setObjectName(QStringLiteral("btnFont_Read"));

        gridLayout->addWidget(btnFont_Read, 12, 3, 1, 1);

        editFont_Out = new QLineEdit(groupBox);
        editFont_Out->setObjectName(QStringLiteral("editFont_Out"));
        editFont_Out->setReadOnly(false);

        gridLayout->addWidget(editFont_Out, 12, 4, 1, 1);

        label_16 = new QLabel(groupBox);
        label_16->setObjectName(QStringLiteral("label_16"));
        label_16->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_16, 13, 0, 1, 1);

        btnColor_In = new QPushButton(groupBox);
        btnColor_In->setObjectName(QStringLiteral("btnColor_In"));
        QPalette palette;
        QBrush brush(QColor(255, 0, 0, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Text, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Text, brush);
        QBrush brush1(QColor(120, 120, 120, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::Text, brush1);
        btnColor_In->setPalette(palette);

        gridLayout->addWidget(btnColor_In, 13, 1, 1, 1);

        btnColor_Write = new QPushButton(groupBox);
        btnColor_Write->setObjectName(QStringLiteral("btnColor_Write"));

        gridLayout->addWidget(btnColor_Write, 13, 2, 1, 1);

        btnColor_Read = new QPushButton(groupBox);
        btnColor_Read->setObjectName(QStringLiteral("btnColor_Read"));

        gridLayout->addWidget(btnColor_Read, 13, 3, 1, 1);

        editColor_Out = new QLineEdit(groupBox);
        editColor_Out->setObjectName(QStringLiteral("editColor_Out"));
        editColor_Out->setReadOnly(false);

        gridLayout->addWidget(editColor_Out, 13, 4, 1, 1);

        label_6 = new QLabel(groupBox);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_6, 4, 0, 1, 1);

        spin_Int32 = new QSpinBox(groupBox);
        spin_Int32->setObjectName(QStringLiteral("spin_Int32"));
        spin_Int32->setMinimum(-2147483647);
        spin_Int32->setMaximum(2147483647);
        spin_Int32->setValue(-4294);

        gridLayout->addWidget(spin_Int32, 4, 1, 1, 1);

        btnInt32_Write = new QPushButton(groupBox);
        btnInt32_Write->setObjectName(QStringLiteral("btnInt32_Write"));

        gridLayout->addWidget(btnInt32_Write, 4, 2, 1, 1);

        btnInt32_Read = new QPushButton(groupBox);
        btnInt32_Read->setObjectName(QStringLiteral("btnInt32_Read"));

        gridLayout->addWidget(btnInt32_Read, 4, 3, 1, 1);

        edit_Int32 = new QLineEdit(groupBox);
        edit_Int32->setObjectName(QStringLiteral("edit_Int32"));
        edit_Int32->setReadOnly(true);

        gridLayout->addWidget(edit_Int32, 4, 4, 1, 1);

        label_7 = new QLabel(groupBox);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_7, 5, 0, 1, 1);

        spin_Int64 = new QSpinBox(groupBox);
        spin_Int64->setObjectName(QStringLiteral("spin_Int64"));
        spin_Int64->setMinimum(-2147483647);
        spin_Int64->setMaximum(2147483647);
        spin_Int64->setValue(25465);

        gridLayout->addWidget(spin_Int64, 5, 1, 1, 1);

        btnInt64_Write = new QPushButton(groupBox);
        btnInt64_Write->setObjectName(QStringLiteral("btnInt64_Write"));

        gridLayout->addWidget(btnInt64_Write, 5, 2, 1, 1);

        btnInt8_Read = new QPushButton(groupBox);
        btnInt8_Read->setObjectName(QStringLiteral("btnInt8_Read"));

        gridLayout->addWidget(btnInt8_Read, 0, 3, 1, 1);

        label_15 = new QLabel(groupBox);
        label_15->setObjectName(QStringLiteral("label_15"));
        label_15->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_15, 12, 0, 1, 1);

        btnFont_In = new QPushButton(groupBox);
        btnFont_In->setObjectName(QStringLiteral("btnFont_In"));

        gridLayout->addWidget(btnFont_In, 12, 1, 1, 1);

        btnFont_Write = new QPushButton(groupBox);
        btnFont_Write->setObjectName(QStringLiteral("btnFont_Write"));

        gridLayout->addWidget(btnFont_Write, 12, 2, 1, 1);

        btnDouble_Write = new QPushButton(groupBox);
        btnDouble_Write->setObjectName(QStringLiteral("btnDouble_Write"));

        gridLayout->addWidget(btnDouble_Write, 9, 2, 1, 1);

        btnDouble_Read = new QPushButton(groupBox);
        btnDouble_Read->setObjectName(QStringLiteral("btnDouble_Read"));

        gridLayout->addWidget(btnDouble_Read, 9, 3, 1, 1);

        editStr_Out = new QLineEdit(groupBox);
        editStr_Out->setObjectName(QStringLiteral("editStr_Out"));

        gridLayout->addWidget(editStr_Out, 11, 4, 1, 1);

        editStr_In = new QLineEdit(groupBox);
        editStr_In->setObjectName(QStringLiteral("editStr_In"));

        gridLayout->addWidget(editStr_In, 11, 1, 1, 1);

        btnInt64_Read = new QPushButton(groupBox);
        btnInt64_Read->setObjectName(QStringLiteral("btnInt64_Read"));

        gridLayout->addWidget(btnInt64_Read, 5, 3, 1, 1);

        edit_Int64 = new QLineEdit(groupBox);
        edit_Int64->setObjectName(QStringLiteral("edit_Int64"));
        edit_Int64->setReadOnly(true);

        gridLayout->addWidget(edit_Int64, 5, 4, 1, 1);

        label_8 = new QLabel(groupBox);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_8, 6, 0, 1, 1);

        spin_Int = new QSpinBox(groupBox);
        spin_Int->setObjectName(QStringLiteral("spin_Int"));
        spin_Int->setMinimum(-2147483647);
        spin_Int->setMaximum(2147483647);
        spin_Int->setValue(5658);

        gridLayout->addWidget(spin_Int, 6, 1, 1, 1);

        btnInt_Write = new QPushButton(groupBox);
        btnInt_Write->setObjectName(QStringLiteral("btnInt_Write"));

        gridLayout->addWidget(btnInt_Write, 6, 2, 1, 1);

        btnInt_Read = new QPushButton(groupBox);
        btnInt_Read->setObjectName(QStringLiteral("btnInt_Read"));

        gridLayout->addWidget(btnInt_Read, 6, 3, 1, 1);

        edit_Int = new QLineEdit(groupBox);
        edit_Int->setObjectName(QStringLiteral("edit_Int"));
        edit_Int->setReadOnly(true);

        gridLayout->addWidget(edit_Int, 6, 4, 1, 1);

        btnUInt8_Write = new QPushButton(groupBox);
        btnUInt8_Write->setObjectName(QStringLiteral("btnUInt8_Write"));

        gridLayout->addWidget(btnUInt8_Write, 1, 2, 1, 1);

        btnUInt8_Read = new QPushButton(groupBox);
        btnUInt8_Read->setObjectName(QStringLiteral("btnUInt8_Read"));

        gridLayout->addWidget(btnUInt8_Read, 1, 3, 1, 1);

        btnInt16_Write = new QPushButton(groupBox);
        btnInt16_Write->setObjectName(QStringLiteral("btnInt16_Write"));

        gridLayout->addWidget(btnInt16_Write, 2, 2, 1, 1);

        btnInt16_Read = new QPushButton(groupBox);
        btnInt16_Read->setObjectName(QStringLiteral("btnInt16_Read"));

        gridLayout->addWidget(btnInt16_Read, 2, 3, 1, 1);

        edit_Int16 = new QLineEdit(groupBox);
        edit_Int16->setObjectName(QStringLiteral("edit_Int16"));
        edit_Int16->setReadOnly(true);

        gridLayout->addWidget(edit_Int16, 2, 4, 1, 1);

        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_5, 3, 0, 1, 1);

        spin_UInt16 = new QSpinBox(groupBox);
        spin_UInt16->setObjectName(QStringLiteral("spin_UInt16"));
        spin_UInt16->setMaximum(65536);
        spin_UInt16->setValue(65222);

        gridLayout->addWidget(spin_UInt16, 3, 1, 1, 1);

        btnUInt16_Write = new QPushButton(groupBox);
        btnUInt16_Write->setObjectName(QStringLiteral("btnUInt16_Write"));

        gridLayout->addWidget(btnUInt16_Write, 3, 2, 1, 1);

        btnUIn16_Read = new QPushButton(groupBox);
        btnUIn16_Read->setObjectName(QStringLiteral("btnUIn16_Read"));

        gridLayout->addWidget(btnUIn16_Read, 3, 3, 1, 1);

        edit_UInt16 = new QLineEdit(groupBox);
        edit_UInt16->setObjectName(QStringLiteral("edit_UInt16"));
        edit_UInt16->setReadOnly(true);

        gridLayout->addWidget(edit_UInt16, 3, 4, 1, 1);

        chkBox_Out = new QCheckBox(groupBox);
        chkBox_Out->setObjectName(QStringLiteral("chkBox_Out"));

        gridLayout->addWidget(chkBox_Out, 7, 4, 1, 1);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_3, 11, 0, 1, 1);

        btnStr_Write = new QPushButton(groupBox);
        btnStr_Write->setObjectName(QStringLiteral("btnStr_Write"));

        gridLayout->addWidget(btnStr_Write, 11, 2, 1, 1);


        verticalLayout->addWidget(groupBox);

        MainWindow->setCentralWidget(centralWidget);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);

        toolBar->addAction(actClearOutput);
        toolBar->addAction(actSaveALL);
        toolBar->addAction(actReadALL);
        toolBar->addSeparator();
        toolBar->addAction(actClose);

        retranslateUi(MainWindow);
        QObject::connect(actClose, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow Application", nullptr));
        actClearOutput->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272\347\274\226\350\276\221\346\241\206\345\205\250\346\270\205\347\251\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actClearOutput->setToolTip(QApplication::translate("MainWindow", "\350\257\273\345\207\272\347\274\226\350\276\221\346\241\206\345\205\250\346\270\205\347\251\272", nullptr));
#endif // QT_NO_TOOLTIP
        actClose->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
        actSaveALL->setText(QApplication::translate("MainWindow", "\350\277\236\347\273\255\345\206\231\345\205\245\346\226\207\344\273\266", nullptr));
#ifndef QT_NO_TOOLTIP
        actSaveALL->setToolTip(QApplication::translate("MainWindow", "\350\277\236\347\273\255\345\206\231\345\205\245\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        actReadALL->setText(QApplication::translate("MainWindow", "\350\277\236\347\273\255\344\273\216\346\226\207\344\273\266\350\257\273\345\217\226", nullptr));
#ifndef QT_NO_TOOLTIP
        actReadALL->setToolTip(QApplication::translate("MainWindow", "\350\277\236\347\273\255\344\273\216\346\226\207\344\273\266\350\257\273\345\217\226", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox_2->setTitle(QString());
        btnFile->setText(QApplication::translate("MainWindow", "\346\265\213\350\257\225\347\224\250\346\226\207\344\273\266", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\345\255\227\350\212\202\345\272\217", nullptr));
        radio_LittleEndian->setText(QApplication::translate("MainWindow", "LittleEndian", nullptr));
        radio_BigEndian->setText(QApplication::translate("MainWindow", "BigEndian", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\346\265\256\347\202\271\346\225\260\347\262\276\345\272\246", nullptr));
        radio_Single->setText(QApplication::translate("MainWindow", "SinglePrecision", nullptr));
        radio_Double->setText(QApplication::translate("MainWindow", "DoublePrecision", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "\346\265\201\347\274\226\347\240\201\346\226\271\345\274\217\350\257\273\345\206\231\346\265\213\350\257\225", nullptr));
        label_13->setText(QApplication::translate("MainWindow", "QString", nullptr));
        editQStr_In->setText(QApplication::translate("MainWindow", "Hello", nullptr));
        btnQStr_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnQStr_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_10->setText(QApplication::translate("MainWindow", "Bool", nullptr));
        chkBox_In->setText(QApplication::translate("MainWindow", "\345\270\203\345\260\224\345\200\274", nullptr));
        btnBool_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnBool_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        btnInt8_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        label->setText(QApplication::translate("MainWindow", "Int8(-128~127)", nullptr));
        btnStr_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_11->setText(QApplication::translate("MainWindow", "Float", nullptr));
        btnFloat_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnFloat_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_12->setText(QApplication::translate("MainWindow", "Double", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "UInt8(0~255)", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "Int16(-32768~32767)", nullptr));
        btnFont_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        editFont_Out->setText(QApplication::translate("MainWindow", "\345\255\227\344\275\223", nullptr));
        label_16->setText(QApplication::translate("MainWindow", "QVariant", nullptr));
        btnColor_In->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\351\242\234\350\211\262", nullptr));
        btnColor_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnColor_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        editColor_Out->setText(QApplication::translate("MainWindow", "\351\242\234\350\211\262", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "Int32", nullptr));
        btnInt32_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnInt32_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "Int64", nullptr));
        btnInt64_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnInt8_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_15->setText(QApplication::translate("MainWindow", "QVariant", nullptr));
        btnFont_In->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\345\255\227\344\275\223", nullptr));
        btnFont_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnDouble_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnDouble_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        editStr_In->setText(QApplication::translate("MainWindow", "Hello", nullptr));
        btnInt64_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "Int(=Int32)", nullptr));
        btnInt_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnInt_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        btnUInt8_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnUInt8_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        btnInt16_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnInt16_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "UInt16(0~65535)", nullptr));
        btnUInt16_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        btnUIn16_Read->setText(QApplication::translate("MainWindow", "\350\257\273\345\207\272", nullptr));
        chkBox_Out->setText(QApplication::translate("MainWindow", "\345\270\203\345\260\224\345\200\274", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "String", nullptr));
        btnStr_Write->setText(QApplication::translate("MainWindow", "\345\206\231\345\205\245", nullptr));
        toolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
