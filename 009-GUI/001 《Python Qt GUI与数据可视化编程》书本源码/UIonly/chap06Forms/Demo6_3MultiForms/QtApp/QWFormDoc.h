#ifndef QWFORMDOC_H
#define QWFORMDOC_H

#include <QWidget>

namespace Ui {
class QWFormDoc;
}

class QWFormDoc : public QWidget
{
    Q_OBJECT

public:
    explicit QWFormDoc(QWidget *parent = 0);
    ~QWFormDoc();

    void    loadFromFile(QString& aFileName);

private slots:
    void on_actOpen_triggered();

    void on_actFont_triggered();

signals:
    void docFileChanged(QString shorFilename);

private:
    Ui::QWFormDoc *ui;
};

#endif // QWFORMDOC_H
