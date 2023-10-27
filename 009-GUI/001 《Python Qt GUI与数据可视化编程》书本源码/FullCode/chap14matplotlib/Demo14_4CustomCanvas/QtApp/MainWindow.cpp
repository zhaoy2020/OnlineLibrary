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


void MainWindow::on_actTightLayout_triggered()
{

}

void MainWindow::on_actSetCursor_triggered()
{

}

void MainWindow::on_actScatterAgain_triggered()
{

}

void MainWindow::on_gBoxHist_toolbar_clicked(bool checked)
{

}

void MainWindow::on_chkBoxHist_ShowHint_clicked(bool checked)
{

}

void MainWindow::on_btnHist_tightLayout_clicked()
{

}

void MainWindow::on_btnHist_redraw_clicked()
{

}

void MainWindow::on_btnStem_redraw_clicked()
{

}

void MainWindow::on_btnStem_tightLayout_clicked()
{

}

void MainWindow::on_chkBoxStem_Analog_clicked(bool checked)
{

}

void MainWindow::on_chkBoxStem_Holder_clicked(bool checked)
{

}

void MainWindow::on_chkBoxStem_Legend_clicked(bool checked)
{

}

void MainWindow::on_btnPie_redraw_clicked()
{

}

void MainWindow::on_btnPie_tightLayout_clicked()
{

}

void MainWindow::on_chkBoxPie_Legend_clicked(bool checked)
{

}

void MainWindow::on_radioPolar_spiral_clicked()
{

}

void MainWindow::on_radioPolar_rose_clicked()
{

}

void MainWindow::on_chkBoxPolar_direction_clicked(bool checked)
{

}

void MainWindow::on_chkBoxPolar_gridOn_clicked(bool checked)
{

}

void MainWindow::on_spinPolar_offset_valueChanged(int arg1)
{

}

void MainWindow::on_btnPolar_tightLayout_clicked()
{

}

void MainWindow::on_btnPolar_rotate_clicked()
{

}

void MainWindow::on_radioFill_Both_clicked()
{

}

void MainWindow::on_btnFill_tightLayout_clicked()
{

}

void MainWindow::on_chkBoxFill_gridLine_clicked(bool checked)
{

}

void MainWindow::on_chkBoxHist_Legend_clicked(bool checked)
{

}

void MainWindow::on_btnPolar_redraw_clicked()
{

}
