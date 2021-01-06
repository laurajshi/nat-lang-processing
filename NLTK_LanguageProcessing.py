#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit

# ## Ch. 1 Language Processing and Python

# In[1]:


#source = https://www.nltk.org/book/ch01.html
#loads all sample text of books
from nltk.draw.dispersion import dispersion_plot
from nltk.book import *
from matplotlib.pyplot import figure


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


# In[9]:


# can use a dispersion plot to determine the location of the word in the text
# text4 = Inaugural Address Corpus
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
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


# In[34]:


#3.1 Frequency Distributions
fdist1 = FreqDist(text1)
print(fdist1)
fdist1.most_common(10)


# In[35]:


fdist1['whale']


# In[36]:


fdist1.plot(25, cumulative = True)


# In[37]:


#hapaxes are words that occur only once
fdist1.hapaxes()


# In[38]:


listHap = fdist1.hapaxes()
len(listHap)


# In[39]:


V = set(text2) #removes duplicates
long_words = [w for w in V if (len(w)) > 15]
sorted(long_words)


# In[40]:


fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])


# In[41]:


#collocation = sequence of words that occur togehter unusually often, e.g. red wine
#resistant to substitution e.g. red wine not equal to maroon wine
list(bigrams(['more', 'is', 'said', 'than', 'done'])) #bigram = extracts adjacent word pairs, bigrams produces generator object that list iterates through


# In[42]:


[len(w) for w in text1]


# In[43]:


fdist = FreqDist(len(w) for w in text1)
print(fdist)
fdist #where key is the length and the corresponding value is the frequency of that length


# In[44]:


fdist.most_common()


# In[45]:


fdist.max()


# In[46]:


fdist[3]+=1


# In[47]:


sorted(w for w in set(text1) if w.endswith('ableness'))


# In[48]:


sorted(w for w in set(text7) if '-' in w and 'index' in w)


# In[49]:


sorted(w for w in set(text3) if w.istitle() and len(w) >10)


# In[50]:


sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)


# In[51]:


len(text1)


# In[52]:


len(set(word.lower() for word in text1)) #convert all to lower to consider e.g. "This" and "this" duplicates


# In[53]:


len(set(word.lower() for word in text1 if word.isalpha()))


# In[54]:


l = [sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8, sent9]
for i in l:
    print(sorted(set(i)))


# ## Ch. 2 Accessing Text Corpora and Lexical Resources

# In[55]:


import nltk
nltk.corpus.gutenberg.fileids()


# In[56]:


emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)


# In[57]:


len(set(emma))


# In[58]:


emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")


# In[59]:


from nltk.corpus import gutenberg
gutenberg.fileids()


# In[60]:


emma = gutenberg.words('austen-emma.txt')


# In[61]:


for f in gutenberg.fileids():
    num_chars = len(gutenberg.raw(f))
    num_words = len(gutenberg.words(f))
    num_sents = len(gutenberg.sents(f))
    num_vocab = len(set(w.lower() for w in gutenberg.words(f)))
    print(round(num_chars / num_words), round(num_words/ num_sents), round(num_words/ num_vocab), f)


# In[62]:


macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences


# In[63]:


macbeth_sentences[1116]


# In[64]:


longest_len = max(len(s) for s in macbeth_sentences)
[s for s in macbeth_sentences if len(s) == longest_len]


# In[65]:


from nltk.corpus import webtext
for f in webtext.fileids():
    print(f, webtext.raw(f)[:65], "....")


# In[66]:


from nltk.corpus import brown
brown.categories()


# In[67]:


gov_text = brown.words(categories = 'government')


# In[68]:


brown.words(fileids = ['cg22'])


# In[69]:


fdist = nltk.FreqDist(w.lower() for w in gov_text)
modals = ['who', 'what', 'where', 'when', 'why']
for m in modals:
    print(m + ':', fdist[m], end = '; ')


# In[70]:


#conditional frequency distribution to iterate through all genres
cfd = nltk.ConditionalFreqDist((genre, word) 
                               for genre in brown.categories()
                               for word in brown.words(categories = genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions = genres, samples = modals) #generates table on frequency of modal by genre


# In[71]:


from nltk.corpus import reuters
reuters.fileids()


# In[72]:


reuters.categories()


# In[73]:


reuters.categories('training/9865')


# In[74]:


reuters.categories(['training/9865', 'training/9880'])


# In[75]:


reuters.fileids('barley')


# In[76]:


reuters.fileids(['barley', 'corn'])


# In[77]:


reuters.words('training/9865')[:14]


# In[78]:


reuters.words(categories = ['barley', 'corn'])


# In[79]:


from nltk.corpus import inaugural
inaugural.fileids()


# In[80]:


#extract the text year by slicing the first 4 elements
dates = [f[:4] for f in inaugural.fileids()]
dates


# In[81]:


from matplotlib.pyplot import figure
#look at conditional frequency distributioons over time
cfd = nltk.ConditionalFreqDist((target, fileid[:4]) 
                               for fileid in inaugural.fileids()
                               for w in inaugural.words(fileid)
                               for target in ['america', 'citizen', 'democracy']
                               if w.lower().startswith(target))
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
cfd.plot()


# In[82]:


from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang, len(word)) 
                               for lang in languages
                               for word in udhr.words(lang + '-Latin1'))
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
cfd.plot(cumulative = True)


# In[83]:


from nltk.corpus import PlaintextCorpusReader
corpus_root = '/Users/laurashi/Desktop/2021_Spring_Cal/Extracurricular/BANG/NLP_BANG/dict'
wordlists = PlaintextCorpusReader(corpus_root, ".*")
wordlists.fileids()


# In[87]:


wordlists.words('test1.txt')


# In[88]:


wordlists.sents()


# In[95]:


wordlists.sents(fileids = 'test1.txt')[2]


# In[100]:


#conditional frequency distribution based on the category (genre of text)
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist( (genre, word) #produces pairs of the genre and word
                                for genre in brown.categories() #outer loop for each genre
                                for word in brown.words(categories = genre))#inner loop for each word in the genre
genre_word = [(genre, word) 
              for genre in ['news', 'romance']
              for word in brown.words(categories = genre)]
len(genre_word)


# In[101]:


genre_word[:4] #front of the set list will have news


# In[105]:


genre_word[-4:] #back of set list will have romance


# In[109]:


cfd = nltk.ConditionalFreqDist(genre_word)
cfd


# In[110]:


cfd.conditions()


# In[115]:


print('news: ', cfd['news'])
print('romance: ', cfd['romance'])


# In[116]:


cfd['romance'].most_common(20)


# In[123]:


cfd['romance']['could'] #index by category then word will give frequency


# ### Plotting and Tabulating Distributions

# In[124]:


from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist((target, fileid[:4])
                               for fileid in inaugural.fileids()
                               for w in inaugural.words(fileid)
                               for target in ['america', 'citizen']
                               if w.lower().startswith(target))


# In[128]:


from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
             'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang, len(word))
                               for lang in languages
                               for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples = range(10), cumulative= True)


# In[137]:


from nltk.corpus import brown
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
cfd = nltk.ConditionalFreqDist((genre, word)
            for genre in ['news', 'romance']
            for word in brown.words(categories = genre))
cfd.tabulate(samples = days)
cfd.plot(samples = days)


# In[138]:


def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams) # [_bigram-condition]


# In[139]:


cfd['living']


# In[148]:


generate_model(cfd, 'living')


# In[158]:


def lexical_diversity(text_in):
    word_count = len(text_in)
    vocab_size = len(set(text_in)) #individual/ no duplicate words
    return (vocab_size / word_count)


# In[159]:


from nltk.corpus import genesis
kjv = genesis.words('english-kjv.txt')
lexical_diversity(kjv)


# In[165]:


def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('man'):
        return word[:-3] + 'men'
    else:
        return word + 's'


# In[166]:


plural('fairy')


# In[167]:


plural('woman')


# In[168]:


plural('fan')


# In[171]:


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


# In[172]:


unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))


# In[174]:


unusual_words(nltk.corpus.nps_chat.words())


# In[179]:


#find fraction of words that are not in the stopwords list
#stopwords = high frequency words such as helper verbs, 'the', 'to', 'also', 'so'
from nltk.corpus import stopwords
stopwords.words('english')
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


# In[181]:


content_fraction(nltk.corpus.reuters.words())


# In[195]:


#solve the word puzzle problem of finding all the words given the number of letters
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
puzzle_letters = nltk.FreqDist('loyrnci')
must_have = 'i'
[w for w in english_vocab if len(w) >=5 and must_have in w and nltk.FreqDist(w) <= puzzle_letters]


# In[197]:


'ironically' in english_vocab


# In[200]:


names = nltk.corpus.names
names.fileids()
male = names.words('male.txt')
female = names.words('female.txt')
[w for w in male if w in female]


# In[203]:


cfd = nltk.ConditionalFreqDist( (file_id, name[-1]) #last letter of the name
                              for file_id in names.fileids()
                              for name in names.words(file_id))
figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
cfd.plot()


# In[204]:


from nltk.corpus import swadesh
swadesh.fileids()


# In[205]:


swadesh.words('en')


# In[207]:


french2english = swadesh.entries(['fr', 'en'])
french2english


# In[208]:


translate = dict(french2english)
translate['jeter']


# In[217]:


translate['je']


# In[222]:


languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']

for i in range(44,48):
    print(swadesh.entries(languages)[i])


# In[225]:


from nltk.corpus import toolbox
toolbox.entries('rotokas.dic')


# In[226]:


from nltk.corpus import wordnet as wn
wn.synsets('motorcar')


# In[227]:


wn.synset('car.n.01').lemma_names() #lemmas are synonymous words


# In[237]:


wn.synset('car.n.01').definition()


# In[239]:


wn.synset('car.n.01').examples()


# In[240]:


wn.synset('car.n.01').lemmas()


# In[241]:


wn.synsets('car') #multiple different meanings for car


# In[242]:


for synset in wn.synsets('car'):
    print(synset.lemma_names())


# In[244]:


wn.lemmas('car')


# In[253]:


for synset in wn.synsets('dish'):
    print(synset.lemma_names(), synset.definition())


# In[246]:


wn.lemmas('dish')


# In[249]:


wn.synsets('dish')


# In[255]:


motorcar = wn.synset('car.n.01')
types = motorcar.hyponyms()
types[0]


# In[257]:


sorted(lemma.name() for synset in types for lemma in synset.lemmas())


# In[258]:


motorcar.hypernyms()


# In[259]:


paths = motorcar.hypernym_paths()
len(paths)


# In[260]:


[synset.name() for synset in paths[0]]


# In[261]:


[synset.name() for synset in paths[1]]


# In[262]:


#most general hypernym ("root" hypernym)
motorcar.root_hypernyms()


# In[263]:


wn.synset('tree.n.01').part_meronyms()


# In[264]:


wn.synset('tree.n.01').substance_meronyms()


# In[265]:


wn.synset('tree.n.01').member_holonyms()


# In[268]:


for s in wn.synsets('mint', wn.NOUN):
    print(s.name() + ": "+ s.definition())


# In[ ]:




