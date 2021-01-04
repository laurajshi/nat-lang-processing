#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit

# ## Ch. 1 Language Processing and Python

# In[1]:


#source = https://www.nltk.org/book/ch01.html
#loads all sample text of books
from nltk.draw.dispersion import dispersion_plot
from nltk.book import *


# In[2]:


text1.concordance("monstrous")


# In[3]:


text1.similar("monstrous")


# In[4]:


#note differences from Melville to Austen w/r/t similarity
text2.similar("monstrous") 


# In[5]:


text2.common_contexts(["monstrous","very"])


# In[6]:


text3.concordance("garment")


# In[7]:


text3.similar("garment")


# In[8]:


text3.common_contexts(["city", "garment"])


# In[35]:


# can use a dispersion plot to determine the location of the word in the text
# text4 = Inaugural Address Corpus
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America", "liberty"])


# In[10]:


text3.generate() #generates random text in the style of the given text


# In[11]:


len(text3)


# In[12]:


#returns the individual words used since duplicaets are collapsed in a set
sorted(set(text3)) 


# In[13]:


len(set(text3)) #number of unique words present


# In[14]:


#percentage of unique words-> "lexical richness"
len(set(text3))/ len(text3) 


# In[15]:


text3.count("smote")


# In[16]:


(text4.count('a')/ len(text4)) * 100 #percent of words that are 'a'


# In[17]:


(text5.count('lol')/ len(text5)) * 100


# In[18]:


text5.concordance("lol")


# In[19]:


def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count, total):
    return (count / total) * 100


# In[20]:


lexical_diversity(text3)


# In[21]:


lexical_diversity(text5)


# In[22]:


percentage(text4.count('a'), len(text4))


# In[23]:


sent3 = ['My', 'name', 'is', 'Jo', 'jo', '.']


# In[24]:


sent3


# In[25]:


sorted(sent3) #sorts alphabetically with symbols then upper case first then lowercase


# In[26]:


sent4 + sent1


# In[27]:


sent1.append("Some")
sent1


# In[28]:


sent = ['word1', 'word2', 'word3', 'word4', 'word5',
        'word6', 'word7', 'word8', 'word9', 'word10']
sent[0]


# In[29]:


sent[0] = 'First'
sent[9] = 'Last'
len(sent)


# In[30]:


sent[1:9] = ['Second', 'Third']
sent


# In[31]:


'Monty . Python'.split(' . ')


# In[32]:


saying = ['After', 'all', 'is', 'said', 'and', 'done',
         'more', 'is', 'said', 'than', 'done']
tokens = set(saying)
tokens = sorted(tokens)
tokens


# In[33]:


tokens[-2:]


# In[39]:


#3.1 Frequency Distributions
fdist1 = FreqDist(text1)
print(fdist1)
fdist1.most_common(10)


# In[40]:


fdist1['whale']


# In[43]:


fdist1.plot(25, cumulative = True)


# In[46]:


#hapaxes are words that occur only once
fdist1.hapaxes()


# In[50]:


listHap = fdist1.hapaxes()
len(listHap)


# In[60]:


V = set(text2) #removes duplicates
long_words = [w for w in V if (len(w)) > 15]
sorted(long_words)


# In[61]:


fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])


# In[64]:


#collocation = sequence of words that occur togehter unusually often, e.g. red wine
#resistant to substitution e.g. red wine not equal to maroon wine
list(bigrams(['more', 'is', 'said', 'than', 'done'])) #bigram = extracts adjacent word pairs, bigrams produces generator object that list iterates through


# In[70]:


[len(w) for w in text1]


# In[73]:


fdist = FreqDist(len(w) for w in text1)
print(fdist)
fdist #where key is the length and the corresponding value is the frequency of that length


# In[74]:


fdist.most_common()


# In[75]:


fdist.max()


# In[76]:


fdist[3]+=1


# In[78]:


sorted(w for w in set(text1) if w.endswith('ableness'))


# In[79]:


sorted(w for w in set(text7) if '-' in w and 'index' in w)


# In[80]:


sorted(w for w in set(text3) if w.istitle() and len(w) >10)


# In[83]:


sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)


# In[84]:


len(text1)


# In[82]:


len(set(word.lower() for word in text1)) #convert all to lower to consider e.g. "This" and "this" duplicates


# In[85]:


len(set(word.lower() for word in text1 if word.isalpha()))


# In[114]:


l = [sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8, sent9]
for i in l:
    print(sorted(set(i)))


# ## Ch. 2 Accessing Text Corpora and Lexical Resources

# In[119]:


import nltk
nltk.corpus.gutenberg.fileids()


# In[121]:


emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)


# In[123]:


len(set(emma))


# In[125]:


emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")


# In[127]:


from nltk.corpus import gutenberg
gutenberg.fileids()


# In[128]:


emma = gutenberg.words('austen-emma.txt')


# In[129]:


for f in gutenberg.fileids():
    num_chars = len(gutenberg.raw(f))
    num_words = len(gutenberg.words(f))
    num_sents = len(gutenberg.sents(f))
    num_vocab = len(set(w.lower() for w in gutenberg.words(f)))
    print(round(num_chars / num_words), round(num_words/ num_sents), round(num_words/ num_vocab), f)


# In[131]:


macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences


# In[133]:


macbeth_sentences[1116]


# In[137]:


longest_len = max(len(s) for s in macbeth_sentences)
[s for s in macbeth_sentences if len(s) == longest_len]


# In[142]:


from nltk.corpus import webtext
for f in webtext.fileids():
    print(f, webtext.raw(f)[:65], "....")


# In[144]:


from nltk.corpus import brown
brown.categories()


# In[147]:


gov_text = brown.words(categories = 'government')


# In[148]:


brown.words(fileids = ['cg22'])


# In[150]:


fdist = nltk.FreqDist(w.lower() for w in gov_text)
modals = ['who', 'what', 'where', 'when', 'why']
for m in modals:
    print(m + ':', fdist[m], end = '; ')


# In[155]:


#conditional frequency distribution to iterate through all genres
cfd = nltk.ConditionalFreqDist((genre, word) 
                               for genre in brown.categories()
                               for word in brown.words(categories = genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions = genres, samples = modals) #generates table on frequency of modal by genre


# In[158]:


from nltk.corpus import reuters
reuters.fileids()


# In[159]:


reuters.categories()


# In[160]:


reuters.categories('training/9865')


# In[161]:


reuters.categories(['training/9865', 'training/9880'])


# In[163]:


reuters.fileids('barley')


# In[164]:


reuters.fileids(['barley', 'corn'])


# In[167]:


reuters.words('training/9865')[:14]


# In[168]:


reuters.words(categories = ['barley', 'corn'])


# In[169]:


from nltk.corpus import inaugural
inaugural.fileids()


# In[170]:


#extract the text year by slicing the first 4 elements
dates = [f[:4] for f in inaugural.fileids()]
dates


# In[181]:


from matplotlib.pyplot import figure
#look at conditional frequency distributioons over time
cfd = nltk.ConditionalFreqDist((target, fileid[:4]) 
                               for fileid in inaugural.fileids()
                               for w in inaugural.words(fileid)
                               for target in ['america', 'citizen', 'democracy']
                               if w.lower().startswith(target))
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
cfd.plot()


# In[ ]:




