#include "MainWindow.h"
#include "ui_MainWindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_mouseMove(QPoint point)
{//鼠标移动事件，point是 GraphicsView的坐标,物理坐标
}

void MainWindow::do_mouseClicked(QPoint point)
{//鼠标单击事件
}

void MainWindow::do_mouseDoubleClick(QPoint point)
{//鼠标双击事件,调用相应的对话框，设置填充颜色、线条颜色或字体
}

void MainWindow::do_keyPress(QKeyEvent *event)
{ //按键事件
}

void MainWindow::on_actItem_Rect_triggered()
{ //添加一个矩形
}

void MainWindow::on_actItem_Ellipse_triggered()
{ //添加一个椭圆
}

void MainWindow::on_actItem_Polygon_triggered()
{ //添加一个梯形
}

void MainWindow::on_actEdit_Delete_triggered()
{ //删除所有选中的绘图项
}

