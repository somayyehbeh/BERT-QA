import pandas as pd
def translate(path='./test.txt'):
	dataframe = pd.read_csv(path, delimiter = "\t")
	dataframe.columns = ['number', 'subject1_mid', 'subject1', 'predicate_class',
						 'subject2_mid', 'Question', 'entity_label']
	dataframe[['Question', 'subject2_mid']].to_excel('test.xlsx', index=False)


if __name__=='__main__':
	translate()