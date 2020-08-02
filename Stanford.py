import nltk
import pandas as pd

file = open(r"C:\Users\Santosh Kumar\Desktop\D20055476-Deed1-L2.txt")
text = file.read().replace("\n", ' ')



#text = 

words = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(words)



nltk.help.upenn_tagset('NNP')
chunks = nltk.ne_chunk(pos_tags,binary = False)


enti= []
labels = []


for chunk in chunks:
    if hasattr(chunk,'label'):
        #print(chunk)
        enti.append(''.join(c[0] for c in chunk))
        labels.append(chunk.label())


enti_labels = list(set(zip(enti,labels)))
enti_df = pd.DataFrame(enti_labels)
enti_df.columns = ["Ent","Labels"]
#print(enti_df)


 #==========
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk import word_tokenize
import os


jar=r'C:\Users\Santosh Kumar\AppData\Local\Programs\Python\Python37\Lib\stanford-ner-tagger\stanford-ner.jar'
model=r"C:\Users\Santosh Kumar\AppData\Local\Programs\Python\Python37\Lib\stanford-ner-tagger\english.all.3class.distsim.crf.ser"

st = StanfordNERTagger(model, jar, encoding='utf-8')

tokenize_text = nltk.word_tokenize(text)
classified_text = st.tag(tokenize_text)
classified_text_df = pd.DataFrame(classified_text)
classified_text_df.drop_duplicates(keep='first',inplace=True)
classified_text_df.reset_index(drop=True,inplace=True)
classified_text_df.columns = ["Entities","Labels"]
#print(classified_text_df)



#=============

tokenize_text = nltk.word_tokenize(text)
classified_text = st.tag(tokenize_text)
netagged_words = tokenize_text
#print(netagged_words)

entities = []
labels = []

from itertools import groupby
for tag ,chunk in groupby(classified_text,lambda x:x[1]):
    if tag =="PERSON":
        entities.append(' '.join(w for w, t in chunk))
        labels.append(tag)

entities_all = list(zip(entities,labels))
print(entities_all)
#entities_unique = list(set(zip(entities,labels)))
#print(entities_unique)
entities_df = pd.DataFrame(entities_all)
#entities_df = pd.DataFrame(entities_unique)
print(entities_df)
#entities_df.columns = ["Entities","Labels"]
#print(entities_df)



