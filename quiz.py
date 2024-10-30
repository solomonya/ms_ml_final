import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

Cs = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]
max_iters = [150, 200, 250, 300]

from hashlib import sha256

name = "Хан Соломон Вячеславович"
sha = sha256(name.encode()).hexdigest()


i_C = 3
i_m = 60
C = Cs[int(sha[i_C], base=16) % 8]
max_iter = max_iters[int(sha[i_m], base=16) % 4]
print((C, max_iter))

test_df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
train_df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
y_train = train_df['label']
X_train = train_df.drop("label", axis=1)
y_test = test_df['label']
X_test = test_df.drop("label", axis=1)

X_train /= 255
X_test /= 255

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)