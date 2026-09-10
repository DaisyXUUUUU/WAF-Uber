# 实验结果说明

这些是公开已完成行程驱动的模拟，不是真实 Uber 的服务率。
验证期选择可能得到零权重，此时两行相同是正确结果。
comparison.csv 合并全部测试日和初始化；paired_differences.csv 是同日同种子的配对差值。
不确定性按日期重采样，先对同日不同初始化求均值；默认仅三天，区间仅供探索，不能作强显著性结论。
平均等待针对已服务乘客，不同策略服务集合可能不同。候选搜索仅改变司机，不改变基线选中的当前订单子集。

Fleet 8: mean daily service-rate change = 0.000 percentage points over 3 test days.
Fleet 12: mean daily service-rate change = 0.000 percentage points over 3 test days.