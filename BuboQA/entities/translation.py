import pandas as pd 

def translate(path='../../data/train.xlsx'):
	dataframe = pd.read_excel(path)
	print(dataframe.tail())


if __name__=='__main__':
	translate()