import matplotlib.pyplot as plt 


def set_plt_default():
    '''Set default for plt.'''
    plt.rcdefaults()


def set_plt_rcParams(
        figsize: tuple= (3, 3), 
        format: str = "svg", 
        **kswargs
    ):
    '''Set configure for plt.'''
    # 设置字体栈（优先级从高到低）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Times New Roman',   # 英文优先使用
        'SimSun',            # 中文宋体
        # 'SimHei',            # 备用中文字体黑体
        # 'Noto Sans CJK SC'   # 最后回退
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['pdf.fonttype'] = 42           # ai可编辑的字体格式
    plt.rcParams['figure.figsize'] = figsize    # figsize
    plt.rcParams['savefig.format'] = format     # svg格式
    plt.rcParams['savefig.transparent'] = True  # 背景是否透明