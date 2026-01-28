from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris = load_iris()
    # print(iris.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names',
    #            'filename', 'data_module'])
    # print("鸢尾花数据集：", iris)
    # print("数据集描述：", iris['DESCR'])
    # print("特征名称：", iris['feature_names'])
    # print("特征数据值：", iris['data'])
    # print("特征数据形状：", iris['data'].shape)
    # print("目标名称：", iris['target_names'])
    # print("目标值：", iris['target'])

    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2)
    print("训练集的特征值：", X_train, X_train.shape)
    print("训练集的目标值：", y_train, len(y_train))
    print("测试集的特征值：", X_test, X_test.shape)
    print("测试集的目标值：", y_test, len(y_test))
