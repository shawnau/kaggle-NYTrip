# Kaggle: New York City Taxi Trip Duration Solution

[Competetion Link](https://www.kaggle.com/c/nyc-taxi-trip-duration)

1. The public LB is involved with some leaked data. See [this discussion](https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/39545)
2. My actual private LB score should be around 0.367.
3. I was intended to use the unleaked data for private LB but overlooked the fact that at least 2 sets of data should be submitted or the system would use the data with the best score, but I just merely submitted one set of data. I was sorry for using the leaked data as my final score.

## 评估标准
RMSLE, 将预测数据做log变换之后再求均方误差

## 数据集
参考data文件夹下说明

## 参考链接
见issues

## 说明
 - `feat`文件夹下存放了生成特征的脚本
 - `model`文件夹下存放了建模的脚本(施工中)
 - `param_config.py`存放了文件路径等配置
 - `ipynb`文件夹下存放了用于分析数据的ipython notebooks
