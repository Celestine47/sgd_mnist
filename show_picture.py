import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

current_path = Path.cwd()

# 直接通过sklearn获取mnist_784数据集
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# 获取mnist的键值
mnist.keys()


# 获取特征与标签
X, y = mnist["data"], mnist["target"]
print(X.shape, y.shape)

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
# 显示灰色图
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
# 显示彩色图
# plt.imshow(some_digit_image)
plt.axis("off")

# 保存灰色图
plt.savefig(Path(current_path, "./images/some_digit_plot.png"), dpi=600)
# 保存彩色图
# plt.savefig(Path(current_path, "./images/some_digit_plot_colour.png"), dpi=600)
plt.show()
