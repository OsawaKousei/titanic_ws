import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# データの読み込み
path = '~/titanic_ws/data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')