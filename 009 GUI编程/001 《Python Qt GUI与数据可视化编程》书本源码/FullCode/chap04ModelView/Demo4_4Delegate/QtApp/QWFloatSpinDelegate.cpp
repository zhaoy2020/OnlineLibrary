#include "QWFloatSpinDelegate.h"

#include  <QDoubleSpinBox>

QWFloatSpinDelegate::QWFloatSpinDelegate(QObject *parent):QStyledItemDelegate(parent)
{

}

void QWFloatSpinDelegate::setRange(qreal minV, qreal maxV)
{

}

void QWFloatSpinDelegate::setDecimals(int digi)
{

}

QWidget *QWFloatSpinDelegate::createEditor(QWidget *parent,
   const QStyleOptionViewItem &option, const QModelIndex &index) const
{

}

void QWFloatSpinDelegate::setEditorData(QWidget *editor,
                      const QModelIndex &index) const
{
}

void QWFloatSpinDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
}

void QWFloatSpinDelegate::updateEditorGeometry(QWidget *editor,
              const QStyleOptionViewItem &option, const QModelIndex &index) const
{
}
