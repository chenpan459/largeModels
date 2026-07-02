# 预训练超参数优化

[hparam_search.py](hparam_search.py) 脚本基于 [附录 D：为训练循环添加增强功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中的扩展训练函数，通过网格搜索寻找最优超参数。

>[!NOTE]
> 本脚本运行时间较长。建议在文件顶部的 `HPARAM_GRID` 字典中减少要探索的超参数组合数量。
