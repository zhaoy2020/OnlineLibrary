import pandas as pd
from pysankey import sankey

pd.options.display.max_rows = 8
df = pd.read_csv(
    'pysankey/fruits.txt', sep=' ', names=['true', 'predicted']
)
colorDict = {
    'apple':'#f71b1b',
    'blueberry':'#1b7ef7',
    'banana':'#f3f71b',
    'lime':'#12e23f',
    'orange':'#f78c1b'
}
sankey(
    df['true'], df['predicted'], aspect=20, colorDict=colorDict,
    fontsize=12, figureName="fruit"
)
# Result is in "fruit.png"