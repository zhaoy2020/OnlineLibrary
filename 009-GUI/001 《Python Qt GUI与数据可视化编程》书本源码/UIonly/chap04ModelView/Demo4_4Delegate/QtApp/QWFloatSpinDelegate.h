#ifndef QWFLOATSPINDELEGATE_H
#define QWFLOATSPINDELEGATE_H

#include    <QObject>
#include    <QWidget>
#include    <QStyledItemDelegate>

class QWFloatSpinDelegate : public QStyledItemDelegate
{
    Q_OBJECT

private:
    qreal   m_min=0,m_max=10000;
    int  m_decimals=2;

public:
    QWFloatSpinDelegate(QObject *parent=0);

    void setRange(qreal minV, qreal maxV);
    void setDecimals(int digi);

//自定义代理组件必须继承以下4个函数
//创建编辑组件
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const Q_DECL_OVERRIDE;

    void setEditorData(QWidget *editor, const QModelIndex &index) const Q_DECL_OVERRIDE;
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const Q_DECL_OVERRIDE;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option,
                              const QModelIndex &index) const Q_DECL_OVERRIDE;
};

#endif // QWFLOATSPINDELEGATE_H
