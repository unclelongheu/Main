# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

classes = ['A', 'B', 'C', 'D', 'E', 'F']
n_class = len(classes)
indices = range(n_class)


# y_pred = ['A', 'B', 'C', 'D', 'E', 'F']
# y_real = ['A', 'B', 'C', 'D', 'E', 'F']
#
# n_sample = len(y_pred)
#
# confusion = np.zeros((n_class, n_class)).astype(int)
#
# for i in range(n_sample):
#     confusion[classes.index(y_real[i])][classes.index(y_pred[i])] += 1


confusion = np.array([[97, 2, 0, 0, 1, 0],
                      [4, 94, 1, 21, 0, 0],
                      [3, 2, 95, 0, 0, 0],
                      [0, 0, 0, 98, 2, 0],
                      [3, 1, 0, 0, 96, 0],
                      [0, 1, 3, 0, 6, 90]])

# 热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.show()
plt.colorbar()

# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, classes)  # 设置横坐标方向，rotation=45为45度倾斜
plt.yticks(indices, classes)

# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title('Confusion matrix')

plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 显示数据
normalize = False
fmt = '.2f' if normalize else 'd'
thresh = confusion.max() / 2.

for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(second_index, first_index, format(confusion[first_index][second_index], fmt),
                 horizontalalignment="center",
                 color="white" if confusion[first_index, second_index] > thresh else "black")

# 显示
plt.show()