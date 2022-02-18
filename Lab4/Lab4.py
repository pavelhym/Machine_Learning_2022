import re
import codecs
import pandas as pd

from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer
from nltk.corpus import stopwords
import numpy as np

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from heapq import nlargest

from operator import itemgetter
import re
import codecs
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer
from nltk.corpus import stopwords
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from heapq import nlargest
from operator import itemgetter
from collections import Counter
from nltk import tokenize

fileObj = codecs.open( "text.txt", "r", "utf_8_sig" )
text = fileObj.read()
fileObj.close()

text = re.sub("\r", " ", text)
text = re.sub("\n", " ", text)
text = re.sub("_", "", text)
text = re.sub("\. ", " . ", text)

#clean the beggining and the end
text =  text[text.find("CHAPTER I .",text.find("CHAPTER I .")+1):text.find("THE END")]

#First





#to lower

text = text.lower()



chapters =  [m.start() for m in re.finditer('chapter', text)]
len(chapters)


text_chapters = pd.DataFrame()
text_chapters['Chapter'] = range(1,len(chapters)+1)
list_for_text = []


for i in range(len(chapters)):
    if i == len(chapters)-1:
        subtext = text[chapters[i]:]
        subtext = re.sub('[^A-Za-z0-9.!?\-]+', ' ', subtext)
        subtext = re.sub("chapter", "", subtext)
        subtext = subtext[subtext.find(" ",1):]
        list_for_text.append(subtext)
        break
    subtext = text[chapters[i]:chapters[i+1]]
    subtext = re.sub('[^A-Za-z0-9.!?\-]+', ' ', subtext)
    subtext = re.sub("chapter", "", subtext)
    subtext = subtext[subtext.find(" ",1):]
    list_for_text.append(subtext)

text_chapters["text"] = list_for_text
list_for_text[11]

tokens = TreebankWordTokenizer().tokenize(text_chapters["text"][0])


#tokenizer
array_for_tokens = []
for i in range(len(text_chapters)):
    array_for_tokens.append(TreebankWordTokenizer().tokenize(text_chapters["text"][i]))


text_chapters['tokens'] = array_for_tokens

#stopwords 
from nltk.corpus import stopwords
stop_words = stopwords.words("english")


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


clear = []
for i in range(len(text_chapters)):
    tokens_ns = [token for token in text_chapters["tokens"][i] if token not in stop_words]
    clear.append([lemmatizer.lemmatize(token) for token in tokens_ns])

text_chapters['tokens_ns'] = clear


#3rd
#TF-IDF IDF inside what?
TF_IDF_list = []
for chapt in range(0,len(text_chapters['tokens_ns'])):
    tokens_alice_nopunct = [token for token in text_chapters['tokens_ns'][chapt] if token not in ["!",".","?"]]
    len_chapter = len(tokens_alice_nopunct)
    counted =  Counter(tokens_alice_nopunct)
    df_counted = pd.DataFrame.from_dict(counted, orient='index').reset_index()
    df_counted = df_counted.rename(columns={'index':'word', 0:'count'})
    df_counted["TF"] = df_counted["count"]/len_chapter
    temp_list = []
    for word in df_counted["word"]:
        idf = 0
        for chapter in text_chapters['tokens_ns']:
            if word in chapter:
                idf+= 1
        temp_list.append(np.log(len(text_chapters['tokens_ns'])/idf))

    df_counted["IDF"] = temp_list
    df_counted["TF-IDF"] = df_counted["IDF"] * df_counted["TF"]

    zip_iterator = zip(df_counted['word'], df_counted["TF-IDF"])
    a_dictionary = dict(zip_iterator)

    TF_IDF_list.append(a_dictionary)
text_chapters["TF-IDF"] = TF_IDF_list


for i in range(0,len(text_chapters['tokens_ns'])):
    #print(dict(sorted(text_chapters["TF-IDF"][i].items(), key = itemgetter(1), reverse = True)[:10]))
    print("Chapter ",i+1," ",nlargest(10, text_chapters["TF-IDF"][i], key = text_chapters["TF-IDF"][i].get))
    print("")




#4rt

full_text_clean = []

for i in range(len(text_chapters)):
    full_text_clean.append(' '.join(text_chapters['tokens_ns'][i]))

full_text_clean = ''.join(full_text_clean)


from nltk import tokenize
all_sentences =  tokenize.sent_tokenize(full_text_clean)
all_sentences = [re.sub("\.", "", i) for i in all_sentences]

def words_alice(all_sentences):

    sent_with_alice = []
    for sent in all_sentences:
        if sent.find("alice") != -1:
            sent_with_alice.append(sent)
    all_alice = ''.join(sent_with_alice)
    tokens_alice = TreebankWordTokenizer().tokenize(all_alice)
    tokens_alice = [token for token in tokens_alice if token not in ["!","alice","?"]]
    return tokens_alice


alice_tokens = words_alice(all_sentences)

alice_tokens =  Counter(alice_tokens)

alice_tokens.most_common(10)