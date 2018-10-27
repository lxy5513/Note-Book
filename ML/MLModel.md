一、监督：{

1.1 分类算法(线性和非线性)：{

​    感知机

​    KNN [k-nearest neighbors algorithm]

​    概率{
​        	朴素贝叶斯（NB）
​        	Logistic Regression（LR）
​        	最大熵MEM（与LR同属于对数线性分类模型）
​    }

​    支持向量机(SVM)

​    决策树(ID3、CART、C4.5)

​    assembly learning{
​       			 Boosting{
​           			 Gradient Boosting{
​               				 GBDT
​                			 xgboost（传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）；xgboost是Gradient Boosting的一种高效系统实现，并不是一种单一算法。）
​           				 }
​           			 AdaBoost
​       		 }   
​        			Bagging{
​            				随机森林
​      			  	}
​       		 Stacking
​    }

​    ……
}

1.2 概率图模型：{
​    	 HMM( hide MM)
   	 MEMM（最大熵马尔科夫）
   	 CRF
​    ……
}

1.3 回归预测：{
   	 线性回归
​    	树回归
​    	Ridge岭回归
   	 Lasso回归
​    ……
}

……  
}

二、非监督：{
​	2.1 聚类：{				

- 基础聚类
  K—mean
  二分k-mean
  K中值聚类
  GMM聚类

- 层次聚类

- 密度聚类

- 谱聚类
  }

  2.2 主题模型:

  {
  ​    pLSA
  ​    LDA隐含狄利克雷分析
  }

2.3 关联分析：{
​    Apriori算法
​    FP-growth算法
}

2.4 降维：{
​    PCA算法
​    SVD算法
​    LDA线性判别分析
​    LLE局部线性嵌入
}

2.5 异常检测：
……
}

三、半监督学习

四、迁移学习