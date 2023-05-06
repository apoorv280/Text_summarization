#import warnings
#warnings.filterwarnings("ignore")
from functions import *


#import datasets
#dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')

#lst_dics = [dic for dic in dataset["train"]]
#dtf = pd.DataFrame(lst_dics).rename(columns={"article":"text", "highlights":"y"})[["text","y"]].head(20000)
#dtf.to_csv("data_summary.csv", index=False)
#dtf.head()

dtf = pd.read_csv("data_summary.csv")
dtf.head()