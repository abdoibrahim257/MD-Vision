# %%
def ngrams(x, n):
    ngrams = []
    for idx, ix in enumerate(x[0:len(x)-n+1]):
        #print(idx, ix)
        gram = x[idx:idx+n]
        ngrams += [gram]
        
    return ngrams

print(ngrams(["large", "chunk", "of", "text"], 3))
ngrams1 = []
#generate random tokens
tokens = ['bear', 'market', 'is', 'coming', 'soon', 'but', 'bull', 'market', 'is', 'already', 'here']
n1 = ngrams(['<t>'] + tokens + ['</t>'], 2)
ngrams1 += n1
print(ngrams1)

# %%
import pandas as pd

def ngrams_from_dic_to_df(ngrams):
    ngrams_dfs = {}
    for ni in range(2, 6):
        columns = [i for i in range(1,ni+1)]
        print(columns)
        ngrams_df = pd.DataFrame(ngrams[ni], columns=columns)
        print(ngrams_df.head())
        ngrams_df['count'] = 0
        print(ngrams_df.head())
        ngrams_df = ngrams_df.groupby(columns).count().sort_values(['count'], ascending=False).reset_index()
        print(ngrams_df.head())
        ngrams_df['probability'] = ngrams_df['count']/ngrams_df['count'].sum() #
        print(ngrams_df)
        ngrams_dfs[ni] = ngrams_df

    return ngrams_dfs

ngrams = {2: [['<t>', 'bear'], ['bear', 'market'], ['market', 'is'], ['is', 'coming'], ['coming', 'soon'], ['soon', 'but'], ['but', 'bull'], ['bull', 'market'], ['market', 'is'], ['is', 'already'], ['already', 'here'], ['here', '</t>']]}
ngrams_from_dic_to_df(ngrams)

# %%
import pandas as pd
import numpy as np
import re

# %%

offsets_list = []
for split in np.array_split(np.arange(3), 2):
    offsets_list += [[split[0], split[-1]+1]]
    
print(offsets_list)

# %%
r = re.compile(r'\d+(?=( cm))')
print(r)

# %%
#read indiana dataset 
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

df3 = pd.DataFrame()
df3 = df2.explode("captions")
df3.explode("captions")
#save df2 in csv 
# split sentence and remove any row that has words that are <= 2
df3 = df3['captions'].map(lambda x: str(x).split())
# remove any row of size <= 2 
df3 = df3[df3.map(len) > 2]



# %%
df3 = pd.DataFrame(df3)

# %%
for index, row in df3.iterrows():
    print(row)
    print('-----------------')
    print(row['captions'])
    print('-----------------')
    if index > 10:
        break

# %%
for index, row in df.iterrows():
    if index == 2722:
        print(row['findings'])
        print(row['impression'])
        break
    # print(row['captions'])
    # break

# %%
df3 = df2.explode("captions")
df3.explode("captions")
#save df2 in csv 
# split sentence and remove any row that has words that are <= 2
df3 = df3['captions'].map(lambda x: str(x).split())
# remove any row of size <= 2 
df3 = df3[df3.map(len) > 2]

df3.to_csv('indiana_reports_cleaned.csv', index=False)
# print(df3.isna().sum())

# %%
newList = df3.values.tolist()
# Convert each inner list to a tuple and add them to a set
unique_reference = set(tuple(x) for x in newList)

# Convert each tuple in the set back to a list
unique_reference = [list(x) for x in unique_reference]

print(unique_reference[0:5])

# %%

t = "SDKMASKDM SDMAKODAOMDKOMASDASD"
print(t.lower())

# %%
import pandas as pd
# read
df = pd.read_parquet('D:/GAM3A/5-Senior02/GP/KENGIC/MIMIC-medical-report/data/train-00000-of-00001-0dc3c7ebb0311aec.parquet')
formatted_df = pd.DataFrame()
#split the text given in to sentences
#remove the following from findings and impression
# any ___
formatted_df['FINDINGS'] = df['FINDINGS'].str.replace(r'___', '', regex = True)
formatted_df['IMPRESSION'] = df['IMPRESSION'].str.replace(r'___', '', regex = True)

# any Dr.
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'Dr.', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'Dr.', '', regex = True)

# any time formats ex: at 12:00 / at floating numbers
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'(at \d{1,2}:\d{1,2})|(\d{1,2}:\d{1,2})', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'(at \d{1,2}:\d{1,2})|(\d{1,2}:\d{1,2})', '', regex = True)

# any p.m/a.m/am/pm
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'( am )|( pm )|( p\.m)|( a\.m)', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'( am )|( pm )|( p\.m)|( a\.m)', '', regex = True)

# remove floating numbers followed by measurements ex: 12.5
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'\d+\.\d+', '', regex = True)
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'\d+\.', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'\d+\.\d+', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'\d+\.', '', regex = True)

#remove any cm mm inch
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'( cm)|( mm)', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'( cm)|( mm)', '', regex = True)

# remove any 1.,2.,3.,etc.
#done in the above step

# remove , =
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r',|=', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r',|=', '', regex = True)

#remove any numbers
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'\d+', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'\d+', '', regex = True)

#remove any \n
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].str.replace(r'\n', '', regex = True)
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].str.replace(r'\n', '', regex = True)

#split each paragraph on .
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].map(lambda x: str(x).split('.'))
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].map(lambda x: str(x).split('.'))

#remove empty strings
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].map(lambda x: [i.split() for i in x if i != ''])
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].map(lambda x: [i.split() for i in x if i != ''])

#check for since, through, by, on,
#make every token a lower case 
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].apply(lambda x: [[word.lower() for word in sentence] for sentence in x])
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].apply(lambda x: [[word.lower() for word in sentence] for sentence in x])


# #remove at ; however, new, from the sentence 
toRemove = ['at', 'however', 'new', 'from',';']
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].apply(lambda x: [[word for word in sentence if word not in toRemove] for sentence in x])
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].apply(lambda x: [[word for word in sentence if word not in toRemove] for sentence in x])


#remove sentence with through, since, submitted, unchanged, compared, comparison, previous, prior,increase, decrease,increased, decreased,
#findings, film, PICC, yesterday, today, SVC, tube,  
toRemoveSentence = ['through', 'since', 'submitted', 'unchanged', 'compared', 'comparison', 'previous', 'prior', 'increase', 'decrease', 'increased', 'decreased', 'findings', 'film', 'picc', 'yesterday', 'today', 'svc', 'tubes']
formatted_df['FINDINGS'] = formatted_df['FINDINGS'].apply(lambda x: [sentence for sentence in x if not any(word in sentence for word in toRemoveSentence)])
formatted_df['IMPRESSION'] = formatted_df['IMPRESSION'].apply(lambda x: [sentence for sentence in x if not any(word in sentence for word in toRemoveSentence)])

finalDf = pd.DataFrame()
finalDf['captions'] = formatted_df['FINDINGS'] + formatted_df['IMPRESSION']

# remove ['as','above'],['status','quo']
toRemoveSentence = ['above', 'quo']
finalDf['captions'] = finalDf['captions'].apply(lambda x: [sentence for sentence in x if (not any(word in sentence for word in toRemoveSentence) and len(sentence) > 2)])

# %%
#split lists to row
new = finalDf.explode('captions')
newList = new['captions'].tolist()
print(len(newList))
# Convert each inner list to a tuple and add them to a set
# print(type(newList))
unique_ref = set()
for x in newList:
    if isinstance(x, list):
        t = tuple(x)
        unique_ref.add(t)

# %%
#convert from set of tuple to list of list
unique_ref = [list(x) for x in unique_ref]
print(len(unique_ref))
print(unique_ref[7440])

# %%
for index, row in new.iterrows():
    if index == 224:
        print("captions: ",row['captions'])
        break

# %%
print(finalDf.shape)

# %%
for index, row in finalDf.iterrows():
    if index == 223:
        print("captions: ",row['captions'])
        break

# %%
count = 0
for index, row in formatted_df.iterrows():
    if r'am' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'am' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
count = 0
for index, row in formatted_df.iterrows():
    if r'since  ' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'since  ' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
count = 0
for index, row in formatted_df.iterrows():
    if r'through  ' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'through  ' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
count = 0
for index, row in formatted_df.iterrows():
    if r'at  ' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'at  ' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
count = 0
for index, row in formatted_df.iterrows():
    if r'PICC' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'PICC' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
count = 0
for index, row in formatted_df.iterrows():
    if r'from  ' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'from  ' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
#print sentences that has "by" in it from the dataframe formatted_df
count = 0
for index, row in formatted_df.iterrows():
    if r'by  ' in (row['FINDINGS']) :
        count+=1
        print('index:', index)
        print("FINDINGS: ",row['FINDINGS'])
        print('-----------------------------------')
        if r'by  ' in (row['IMPRESSION']):
            count+=1
            print("IMPRESSION: ",row['IMPRESSION'])
            print('===================================')
        
print(count)

# %%
for index, row in formatted_df.iterrows():
    if index == 9:
        print("FININDINS:", row['FINDINGS'])
        print("IMPRESSION:", row['IMPRESSION'])
        break

# # %%
# #bleu score for a single sentence
# from nltk.translate.bleu_score import sentence_bleu

# reference = [['this', 'is', 'a', 'test'], ['this', 'is', 'a', 'test']]
# candidate = ['this', 'are', 'test']

# score = sentence_bleu(reference, candidate)
# print(score)


