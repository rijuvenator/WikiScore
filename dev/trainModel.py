import subprocess as bash
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

import wikipedia as WP
import pickle
import glob
import json
import logging

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
SMALL_NUMS  = set([str(n) for n in range(11)])

STOP_WORDS.update(['also', '00', '000', 'one', 'two'])
STOP_WORDS.update(SMALL_NUMS)
STOP_WORDS.update(map(str, list(range(11, 21))))

productiveSuffixes = [
    'ical',
    'ability',
    'able',
    'age',
    'ally',
    'ance',
    'ant',
    'arian',
    'cracy',
    'dom',
    'escent',
    'ette',
    'esque',
    'fix',
    'full?',
    'gaze',
    'gate',
    'ic',
    'ie',
    'ification',
    'ify',
    'in',
    'ish',
    'itude',
    'ity',
    'less',
    'let',
    'like',
    'ly',
    'ment',
    'ness',
    'oid',
    'ology',
    'ous',
    'phile',
    'punk',
    'tion',
    'wise',
]

productivePrefixes = [
    'anti',
    'auto',
    'cis',
    'co',
    'crypto',
    'de',
    'down',
    'ex',
    'extra',
    'hyper',
    'hypo',
    'inter',
    'mega',
    'micro',
    'mid',
    'mis',
    'non',
    'out',
    'over',
    'post',
    'pre',
    'pro',
    'pseudo',
    'quasi',
    're',
    'retro',
    'semi',
    'sub',
    'super',
    'techno',
    'tele',
    'trans',
    'ultra',
    'un',
    'under',
    'up',
]

suffixRE = re.compile(r'\w+({})$'.format('|'.join(productiveSuffixes)))
prefixRE = re.compile(r'({})\w+$'.format('|'.join(productivePrefixes)))

PS = nltk.stem.PorterStemmer()

SIA = SentimentIntensityAnalyzer()

df = pd.read_pickle('BigPickle_1.pkl')
df = pd.concat([df, pd.read_pickle('BigPickle_13.pkl')])
df = pd.concat([df, pd.read_pickle('BigPickle_2.pkl')])

nDocuments = 10000
subFrame = pd.DataFrame(df[~df.Class_talk.isnull()].sample(nDocuments))
subFrame.reset_index(inplace=True)
del df
subFrame = subFrame.drop(columns=['URL', 'Classes', 'Class', 'title'])
subFrame = subFrame.replace({'Class_talk':{'A':'GA'}})
nDocuments = subFrame.shape[0]

print(subFrame.Class_talk.value_counts())
print( ((subFrame.Class_talk.value_counts())/(subFrame.Class_talk.value_counts().sum())) .apply(lambda x: x**2.) . sum() )

def isClean(word):
    if    word in ',.\'\'``();:%$-–[]'   \
       or word in ('\'s', '\'t')         \
       or word in SMALL_NUMS             \
       or word in STOP_WORDS:
        return False
    return True

def hasAffix(word):
    return bool(re.match(prefixRE, word)) or bool(re.match(suffixRE, word))

def safeDivide(a, b):
    if b != 0:
        return a/b
    return 0

def doWork(text):
    nSents   = 0
    nWords   = 0
    nAffixes = 0
    nWChars  = 0

    Sentiments = {'Doc' : SIA.polarity_scores(text)}
    Sentiments['Sent'] = {key:0. for key in Sentiments['Doc']}

    wordLengths = []
    for sent in nltk.sent_tokenize(text):

        sentimentThisSent = SIA.polarity_scores(sent)
        for key in sentimentThisSent:
            Sentiments['Sent'][key] += sentimentThisSent[key]

        #wordList = nltk.word_tokenize(sent)
        #loweredWordList = [ token.lower() for token in wordList ]
        #cleanWordList = [ word for word in wordList if isClean(word) ]
        #del loweredWordList
        #del wordList

        #cleanWordList = [ token.lower() for token in nltk.word_tokenize(sent) if isClean(token.lower()) ]

        nWordsThisSent  = 0
        nAffixesThisSent  = 0
        nWCharsThisSent = 0
        for token in nltk.word_tokenize(sent):
            if isClean(token.lower()):
                word = token.lower()
                nWordsThisSent += 1
                nWCharsThisSent += len(word)
                wordLengths.append(len(word))
                if hasAffix(word):
                    nAffixesThisSent += 1

        nSents   += 1
        nWords   += nWordsThisSent
        nAffixes += nAffixesThisSent
        nWChars  += nWCharsThisSent

    nSemi  = text.count(';')
    nColon = text.count(':')

    fWordLength = safeDivide(nWChars, nWords)

    Sentiments['Sent'] = {key:safeDivide(Sentiments['Sent'][key], nSents) for key in Sentiments['Sent']}

    binnedResults = np.histogram(wordLengths, bins=10, range=(0., 20.))[0].tolist()

    sentimentList = [Sentiments['Doc' ][key] for key in ('pos', 'neg', 'neu', 'compound')] + \
                    [Sentiments['Sent'][key] for key in ('pos', 'neg', 'neu', 'compound')]

    return [nSents, nWords, nSemi, nColon, nAffixes, fWordLength] + binnedResults + sentimentList


subFrame['nWikiLinks'    ] = subFrame['nWikiLinks'    ].astype('int64')
subFrame['nExternalLinks'] = subFrame['nExternalLinks'].astype('int64')

#subFrame['Sents'         ] = subFrame['Text'          ].apply(getSents)
#subFrame['CWords'        ] = subFrame['Sents'         ].apply(getCleanWords)
#subFrame['Words'         ] = subFrame['CWords'        ].apply(getStemmedWords)
#subFrame['nSents'        ] = subFrame['Sents'         ].apply(len)
#subFrame['nWords'        ] = subFrame['Words'         ].apply(len)
#subFrame['nSemi'         ] = subFrame['Sents'         ].apply(getNSemi)
#subFrame['nColon'        ] = subFrame['Sents'         ].apply(getNColon)
#subFrame['WordLength'    ] = subFrame['CWords'        ].apply(getWordLengths)
#subFrame['nAffixes'      ] = subFrame['CWords'        ].apply(getNAffixes)

derivedQuantities = [
    'nSents',
    'nWords',
    'nSemi',
    'nColon',
    'nAffixes',
    'fWordLength'
]

binnedList = ['{:02d}_{:02d}'.format(i, i+2) for i in range(0, 20, 2)]

derivedQuantities.extend(['nWords_'+b for b in binnedList])

derivedQuantities.extend(['nDocSenti_'+s for s in ('pos', 'neg', 'neu', 'compound')])
derivedQuantities.extend(['nSenSenti_'+s for s in ('pos', 'neg', 'neu', 'compound')])

for name, val in zip(derivedQuantities, list(zip(*subFrame['Text'].apply(doWork)))):
    subFrame[name] = val

for name in ['nWikiLinks', 'nExternalLinks'] + derivedQuantities:
    if name in ('fWordLength', 'nSents'): continue
    if 'Senti_' in name: continue
    subFrame['f'+name[1:]] = subFrame[name] / subFrame['nSents']

#for r, l in zip(binnedList, list(zip(*subFrame['CWords'].apply(getNWordsByCounts)))):
#    subFrame['nWords_'+r] = l
#subFrame['nWords_00_05'], subFrame['nWords_05_10'], subFrame['nWords_10_15'], subFrame['nWords_15_20'] = zip(*subFrame['CWords'].apply(getNWordsByCounts))
#subFrame['nWords_00_05'], subFrame['nWords_05_10'], subFrame['nWords_10_15'], subFrame['nWords_15_20'] = zip(*subFrame['CWords'].apply(getNWordsByCounts))

#subFrame['fWords'        ] = subFrame['nWords'        ] / subFrame['nSents']
#subFrame['fSemi'         ] = subFrame['nSemi'         ] / subFrame['nSents']
#subFrame['fColon'        ] = subFrame['nColon'        ] / subFrame['nSents']
#subFrame['fWikiLinks'    ] = subFrame['nWikiLinks'    ] / subFrame['nSents']
#subFrame['fExternalLinks'] = subFrame['nExternalLinks'] / subFrame['nSents']
#subFrame['fAffixes'      ] = subFrame['nAffixes'      ] / subFrame['nSents']
#
#for r in rrList:
#    subFrame['fWords_'+r] = subFrame['nWords_'+r] / subFrame['nSents']
#
#subFrame = subFrame.drop(columns=['Sents', 'CWords'])

print(subFrame[['Title', 'Class_talk', 'Text', 'nSents', 'nWords', 'nExternalLinks', 'nSemi', 'fWordLength', 'nDocSenti_neu']].head())
print(subFrame[['nWords_'+r for r in binnedList]].head())

#h = subFrame.hist('WordLength', by='Class_talk')
#plt.show()
#input()

subFrame.columns

featureNames = [s for s in subFrame.columns if re.match(r'(n|f)[A-Z]\w+', s)]
featureNames.extend(['nDocSenti_'+s for s in ('pos', 'neg', 'neu', 'compound')] + ['nSenSenti_'+s for s in ('pos', 'neg', 'neu', 'compound')])
#ExtraFeatureList = ['nWikiLinks', 'nExternalLinks', 'nWords', 'nSents', 'nSemi', 'fWords', 'fSemi', 'fWikiLinks', 'fExternalLinks', 'WordLength', 'nAffixes', 'fAffixes', 'nColon', 'fColon']
#ExtraFeatureList.extend(['nWords_'+r for r in rrList])
#ExtraFeatureList.extend(['fWords_'+r for r in rrList])
featurelist = []
for i in range(subFrame.shape[0]):
    if subFrame['Class_talk'][i] is None: continue
    #dd = subFrame['Features'][i]
    dd = {}
    dd.update({key:subFrame[key][i] for key in featureNames})
    featurelist.append((dd, subFrame['Class_talk'][i]))

del subFrame

#train_set, test_set = featurelist[:len(featurelist)//2], featurelist[len(featurelist)//2:]
np.random.shuffle(featurelist)
train_set, test_set = sklearn.model_selection.train_test_split(featurelist, train_size=0.8)

real_train_set = train_set

classifier = nltk.NaiveBayesClassifier.train(real_train_set)
print('Accuracy:', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
print('Accuracy on Training Data:', nltk.classify.accuracy(classifier, train_set))

pickle.dump(classifier, open('model.pkl', 'wb'))

exit()

#class Document():
#    def __init__(self, i):
#        self.index      = i
#        self.doc        = subFrame['Text'][i]
#        self.sents      = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(self.doc)]
#        
#        self.wordLists  = [ [token.lower() for token in sent] for sent in self.sents ]
#        
#        self.CWordLists = [ [word for word in wordList if self.isClean(word)] for wordList in self.wordLists ]
#        
#        self.words      = [ word for wordList in self.CWordLists for word in wordList ]
#        
#        self.Class      = subFrame['Class_talk'][i]
#
#        self.nSent      = len(self.sents)
#        self.nWords     = len(self.words)
#        
#        self.nSemi      = sum([sent.count(';') for sent in self.sents])
#        
#        self.nWLinks    = int(subFrame['nWikiLinks'][i])
#        self.nELinks    = int(subFrame['nExternalLinks'][i])
#
#    def isClean(self, word):
#        if    word in ',.\'\'``();:%$-–[]'   \
#           or word in ('\'s', '\'t', 'also') \
#           or word in SMALL_NUMS              \
#           or word in STOP_WORDS:
#            return False
#        return True

def getSents(x):
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(x)]

def getCleanWords(x):
    wordList = [ token.lower() for sent in x for token in sent ]
    return [ word for word in wordList if isClean(word) ]
    #[ 
    #    PS.stem(word) for wordList in
    #        [ 
    #            [ word for word in wordList if isClean(word) ]
    #                for wordList in
    #                    [
    #                        [ token.lower() for token in sent ]
    #                            for sent in x
    #                    ]
    #        ]
    #            for word in wordList
    #]

def getStemmedWords(x):
    #return [ PS.stem(word) for word in x ]
    return x

def getNSemi(x):
    return sum([s.count(';') for s in x])

def getNColon(x):
    return sum([s.count(':') for s in x])

def getWordLengths(x):
    return safeDivide(sum([len(word) for word in x]), len(x))

def getNWordsByCounts(x):
    l = list(map(len, x))
    r = np.histogram(l, bins=10, range=(0., 20.))[0].tolist()
    return r
    #return len([word for word in x if nLow <= len(word) <= nHigh])

def getNAffixes(x):
    return sum([hasAffix(word) for word in x])

#balance = False
#if balance:
#    cls = [cl for i, cl in train_set]
#    counts = {c:cls.count(c) for c in set(cls) if c != 'A'}
#    mSize = min(counts.values())
#    for c in counts:
#        counts[c] = 0
#
#    real_train_set = []
#    for f, c in train_set:
#        if c == 'A': continue
#        if counts[c] < mSize//2:
#            real_train_set.append((f, c))
#            counts[c] += 1
#
#    print('mSize = {}, so the next line should be {}'.format(mSize, mSize//2*6))
#    print(len(real_train_set))
#else:
#    real_train_set = train_set




subFrame = subFrame.drop(columns=['Words'])

#VR = sklearn.feature_extraction.text.TfidfVectorizer(min_df=10, max_df=50, stop_words=newStopWords)
#TFIDFMatrix = VR.fit_transform(subFrame['Text'])
#TFIDFMatrix = VR.fit_transform(subFrame['Words'].apply(lambda x: ' '.join(x)))
#Features = VR.get_feature_names()
#BoW = pd.Series(TFIDFMatrix.toarray().tolist())


#def getCommonFeatures(features, matrix):
#    scores = list(zip(features, np.asarray(matrix.sum(axis=0)).ravel()))
#    scores.sort(key=lambda x: x[1], reverse=True)
#    return set([f for f,v in scores[:2000]])
#
#CommonFeatures = getCommonFeatures(Features, TFIDFMatrix)
#
#def SparseToDict(x):
#    dd = dict( zip( Features, x ) )
#    return {d:v for d,v in dd.items() if d in CommonFeatures}
#
#subFrame['Features'      ] = BoW.apply(SparseToDict)

