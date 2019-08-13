submit记录比赛代码，gnn为尝试使用SR-GNN进行建模(对代码进行了重构，方便检查)
SR-GNN论文地址:https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/3804

本目录下的代码仅为测试，以submit下的代码为准

record记录分数、思路、思考方式、遇到的问题和解决方法，重要！必读！

visualize为部分可视化代码，剩余可视化代码未能保存(已丢失)

offline为离线测试代码，版本原始，准确度不理想，可以通过cv提高准确度(尚未完成)

calc_score为计算离线测试和真实数据之间的分数

run为代码主体，以submit下的代码为最新版本

gnn文件夹中记录使用SR-GNN进行处理的代码，先运行run_session-gnn.py生成临时文件，再运行run_sr-gnn_model.py计算结果，run_sr-gnn_model.py中使用tensorflow-gpu运行，部分参考自https://github.com/CRIPAC-DIG/SR-GNN/tree/master/tensorflow_code，由论文作者github提供

简单总结：
使用简单的协同过滤和规则填充(用户有明显的规律性)即可达到一定的成绩。但是再继续改进则有难度。

容易召回，难于排序，原因：由于缺失用户数据和只有购买行为记录(没有点击、浏览、收藏、购物车等其他行为记录)，构建负样本颇有难度，构建负样本时，一个userID对应一个itemID，但是userID没有特征，只有itemID的特征和user与item之间的交互特征，而且通过可视化分析发现99%以上的用户都有回购行为(即重复购买以前购买过的item)，user与item之间的交互特征容易出现极端情况，预测集中的itemID在item数据中大部分缺失，即使成功构建了训练集，也很难构建有效的预测集，因此传统排序方法较难应用(xgboost等)。
目前只完成了负样本中userID和itemID的对应关系的搜索(优先从user购买过的item的cate和store的交集中随机取一个作为负样本，其次从store中随机取，再其次从cate随机取，最后从所有item中随机取，取的负样本不能在用户的已购买item里)，尚未完成负样本中的特征构建，预计设想user与item之间的交互特征为用户在购买一个item之前的行为的统计特征，负样本的选取可以考虑进行改进：优先从user购买过的item的cate和store和neighbor中的交集中随机取一个作为负样本，其次从store和neighbor中随机取，再其次从cate和neighbor随机取，最后从所有neighbor随机取，取的负样本不能在用户的已购买item里。要注意选取的样本不要最终是正样本，否则模型在做倒功。

在google学术上找到SR-GNN，原理参考论文，但是预测效果不好，估计原因：1、参数设置不当，可以考虑放宽参数(item最少出现次数原先为2，应该提高；剔除购买商品数量超过50的用户，可以适当提高)；2、用户回购行为过多(同一个item重复购买3次、5次甚至全部)是否会影响SR-GNN的训练？