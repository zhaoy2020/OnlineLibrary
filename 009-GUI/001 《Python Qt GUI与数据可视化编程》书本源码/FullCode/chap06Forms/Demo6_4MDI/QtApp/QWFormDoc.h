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


private:
    Ui::QWFormDoc *ui;

};

#endif // QWFORMDOC_H
