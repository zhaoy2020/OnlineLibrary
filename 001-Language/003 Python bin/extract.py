# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 18:26:02 2021

提取序列模块

@author: Bio-windows
"""


from Bio import SeqIO


def extract(db_handle, extract_list):
    '''从fasta数据库中，按要求提取序列
        db_handle:数据库的位置
        extract_list:要提取的序列列表,dataframe格式
        函数最终返回的的含有SeqRecord类的列表
    '''
    # print('需提取序列...\n {}'.format(extract_list.values))

    store = []  # 临时存放Bio.SeqRecord对象

    num = 0  # 计数

    for rec in SeqIO.parse(db_handle, format='fasta'):
        # for rec_extr in extract_list[0].values:
        for rec_extr in extract_list:
            if rec.id == rec_extr:  # 一定要用id，而不是name，虽然内容是一样的，但到SeqIO.write（）时候只能识别id
                num = num + 1  # 序号和计数
                print('正在提取', rec.name, rec.description, num)
                if rec.seq:  # 判断rec是否为空，不为空则添加到store列表中，表示成功提取
                    store.append(rec)
                    print('成功',len(rec.seq),'bp')
                else:
                    print('序列为空！')
    if store:
        print('提取完成\n')
    else:
        print('无对应序列，提取失败！！！\n')
    return store


if __name__ == '__main__':
    '''main function'''
# =============================================================================
#     tem_faa = extract('RawDatas/pan_genome_reference.fa', locals()['unique_'+i][i])
#     SeqIO.write(tem_faa, 'results/unique_'+i+'.faa', 'fasta')
# =============================================================================
# =============================================================================
#         tem_faa = []
#         timer = 1
#         for rec in SeqIO.parse(pan_genome_reference_handle, format='fasta'):
#             for unique_rec in locals()['unique_'+i][i]:
#                 if rec.id == unique_rec:
#                     tem_faa.append(rec)
#                     print('提取 {} {}'.format(unique_rec, timer))
#                     timer = timer + 1
# =============================================================================
