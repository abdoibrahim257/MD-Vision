#read indiana dataset
import pandas as pd
import re

 
df = pd.read_csv('./indiana_reports.csv')

#to capture
measurement = re.compile(r'(\d+(.\d+)?)( )?((cm|mm)?( )?(x) (\d+(.\d+)?) )?(cm|mm)')
ratio = re.compile(r'(\d+(.\d+)\/)')
rankNumbers = re.compile(r'[0-9](st|nd|rd|th)', re.I)
words = re.compile(r'(day|film|recommend|prior|comparison|compare|image|T6|T8|T11|T12)', re.I)
intact = re.compile(r'((?<= )( )?(is|are) intact)|((?<=  )(is|are) unremarkable)')
#create a new dataframe df2
df2 = pd.DataFrame()
# df2['uid'] = df['uid']

df['findings'] = df['findings'].str.replace(r'XXXX', '',regex = True)
df['impression'] = df['impression'].str.replace(r'(XXXX\.|XXXX)', '',regex = True)

#removing any 2.0 cm or 2.0 x 2.0 cm or 2.0 cm or 2.0mm or 2.0 x 2.0mm or 2.0mm x 2.0mm etc.
# df['findings'] = df['findings'].str.replace(r'(\d+(.\d+)?)( )?((cm|mm)?( )?(x) (\d+(.\d+)?) )?(cm|mm)', '',regex=True)
# df['impression'] = df['impression'].str.replace(r'(\d+(.\d+)?)( )?((cm|mm)?( )?(x) (\d+(.\d+)?) )?(cm|mm)', '',regex=True)

#remove any list numbu2. 3. and so on
df['findings'] = df['findings'].str.replace(r'([0-9](\.))|(^[0-9](\.))', '',regex=True)
df['impression'] = df['impression'].str.replace(r'([0-9](\.))|(^[0-9](\.))', '',regex=True) 

#remove comas
df['findings'] = df['findings'].str.replace(r',|-', '',regex=True)
df['impression'] = df['impression'].str.replace(r',|-', '',regex=True)

#split each to plst of sentences 
df['findings'] = df['findings'].map(lambda x: str(x).split('.'))
df['impression'] = df['impression'].map(lambda x: str(x).split('.'))


df['findings'] = df['findings'].apply(lambda x: [sentence for sentence in x if not (intact.search(sentence) or words.search(sentence) or rankNumbers.search(sentence) or measurement.search(sentence) or ratio.search(sentence))])
df['impression'] = df['impression'].apply(lambda x: [sentence for sentence in x if not (intact.search(sentence) or words.search(sentence) or rankNumbers.search(sentence) or measurement.search(sentence) or ratio.search(sentence))])

#loop on the each senctence in the list of sentences and remove any remaining numbers from the sentence 
df['findings'] = df['findings'].apply(lambda x: [re.sub(r'\d+', '', sentence) for sentence in x])
df['impression'] = df['impression'].apply(lambda x: [re.sub(r'\d+', '', sentence) for sentence in x])

#remove any empty sentences
df['findings'] = df['findings'].apply(lambda x: [sentence for sentence in x if sentence.strip()])
df['impression'] = df['impression'].apply(lambda x: [sentence for sentence in x if sentence.strip()])

df2['captions'] = df['findings'] + df['impression']


df3 = df2.explode("captions")
df3.explode("captions")
#save df2 in csv 
# split sentence and remove any row that has words that are <= 2
df3 = df3['captions'].map(lambda x: str(x).split())
# remove any row of size <= 2 
df3 = df3[df3.map(len) > 2]

data = pd.DataFrame(df3)

print(data)
print(df3)