import matplotlib.pyplot as plt 


def set_plt_default() -> None:
    '''
    Set default for plt.
    Demo:
    >>>set_plt_default()
    '''
    plt.rcdefaults()


def set_plt_rcParams(
        figsize: tuple= (3, 3), 
        format: str = "svg", 
        **kswargs
    ) -> None:
    '''
    Set configure for plt.
    Demo:
    >>>set_plt_rcParams()
    >>>set_plt_rcParams()
    '''

    # 设置字体栈（优先级从高到低）
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Times New Roman',   # 英文优先使用
        'SimSun',            # 中文宋体
    ]
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['pdf.fonttype'] = 42           # ai可编辑的字体格式
    plt.rcParams['figure.figsize'] = figsize    # figsize
    plt.rcParams['savefig.format'] = format     # svg格式
    plt.rcParams['savefig.transparent'] = True  # 背景是否透明