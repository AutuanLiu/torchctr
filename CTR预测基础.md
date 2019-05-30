# CTR 预测基础

在计算广告和推荐系统中，CTR预估(click-through rate)是非常重要的一个环节，判断一个商品的是否进行推荐需要根据CTR预估的点击率来进行。在进行
CTR预估时，除了单特征外，往往要对特征进行组合。对于特征组合来说，业界现在通用的做法主要有两大类：**FM系列与Tree系列**

FM(Factorization Machine)主要是为了解决**数据稀疏**的情况下，特征怎样组合的问题。普通的线性模型，我们都是将各个特征独立考虑的，并没有考虑
到特征与特征之间的相互关系。但实际上，大量的特征之间是有关联的。一般的线性模型压根没有考虑特征间的关联。为了表述特征间的相关性，我们采用**多项式模型**。与线性模型相比，FM的模型就多了后面**特征组合**的部分。



## 参考文献

1. [推荐系统遇上深度学习(一)--FM模型理论和实践 - 简书](https://www.jianshu.com/p/152ae633fb00)
2. [简单易学的机器学习算法——因子分解机(Factorization Machine) - null的专栏 - CSDN博客](https://blog.csdn.net/google19890102/article/details/45532745)
3. [分解机(Factorization Machines)推荐算法原理 - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/6370127.html)
4. [机器学习算法系列（26）：因子分解机（FM）与场感知分解机（FFM） | Free Will](https://plushunter.github.io/2017/07/13/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%EF%BC%8826%EF%BC%89%EF%BC%9A%E5%9B%A0%E5%AD%90%E5%88%86%E8%A7%A3%E6%9C%BA%EF%BC%88FM%EF%BC%89%E4%B8%8E%E5%9C%BA%E6%84%9F%E7%9F%A5%E5%88%86%E8%A7%A3%E6%9C%BA%EF%BC%88FFM%EF%BC%89/)
5. [第09章：深入浅出ML之Factorization家族 | 计算广告与机器学习](http://www.52caml.com/head_first_ml/ml-chapter9-factorization-family/)
6. [深入FFM原理与实践 - 美团技术团队](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)
7. [从FFM到DeepFFM，推荐排序模型到底哪家强？](https://www.infoq.cn/article/vKoKh_ZDXcWRh8fLSsRp)
8. [FM与FFM的区别 - AI_盲的博客 - CSDN博客](https://blog.csdn.net/xwd18280820053/article/details/77529274)
9. [矩阵分解在推荐系统中的应用：NMF和经典SVD实战 | 乐天的个人网站](https://www.letiantian.me/2015-05-25-nmf-svd-recommend/)
10. [TF-IDF与余弦相似度 - 知乎](https://zhuanlan.zhihu.com/p/32826433)
11. [王喆的机器学习笔记 - 知乎](https://zhuanlan.zhihu.com/wangzhenotes)
12. [Embedding在深度推荐系统中的3大应用方向 - 知乎](https://zhuanlan.zhihu.com/p/67218758)
13. [谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】 - 知乎](https://zhuanlan.zhihu.com/p/63186101)