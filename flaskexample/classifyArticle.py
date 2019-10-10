import requests
import mwparserfromhell
import subprocess as bash
import json
import pickle
import os
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

FLASK_PREFIX = 'flaskexample/static/'

COLORS     = {'FA': '9CBDFF', 'GA': '66FF66', 'B': 'B2FF66', 'C': 'FFFF66', 'START': 'FFAA66', 'STUB':'FFA4A4'}
TEXTCOLORS = {'FA': '6d84b2', 'GA': '47b247', 'B': '7db247', 'C': 'fddd00', 'START': 'ff954f', 'STUB':'ff1717'}

DEFAULT     = {'title':'-', 'error':'ERROR'}
FNULL       = open(os.devnull, 'w')
STOP_WORDS  = set(nltk.corpus.stopwords.words("english"))
SMALL_NUMS  = set([str(n) for n in range(11)])
CLASSIFIER  = pickle.load(open(FLASK_PREFIX + 'model.pkl', 'rb'))
SIA         = SentimentIntensityAnalyzer()

STOP_WORDS.update(['also', '00', '000', 'one', 'two'])
STOP_WORDS.update(SMALL_NUMS)
STOP_WORDS.update(map(str, list(range(11, 21))))

ProductiveSuffixes = [
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

ProductivePrefixes = [
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

FlavorText = {
        'FA':' a <b>featured article</b>.<br><br>Featured articles exemplify Wikipedia\' best work and are distinguished by professional standards of writing, presentation, and sourcing, in addition to meeting all of Wikipedia\'s <a class="normallink" href="https://en.wikipedia.org/wiki/Wikipedia:Core_content_policies">content policies</a>.<br><br>',
        'GA':' a <b>good article</b>.<br><br>Good articles meet all of Wikipedia\'s <a class="normallink" href="https://en.wikipedia.org/wiki/Wikipedia:Core_content_policies">content policies</a> and provide a well-written, clear and complete description of the topic; only minor style issues and other details need to be addressed before submission as a featured article candidate.<br><br>',
        'B':' <b>B-class</b>.<br><br>B-class articles are mostly complete and without major problems, are reasonably well-written but not necessarily &quot;brilliant&quot;, provide reasonable coverage of the topic, are suitably referenced, and meet Wikipedia\'s <a class="normallink" href="https://en.wikipedia.org/wiki/Wikipedia:Core_content_policies">content policies</a>.<br><br>',
    'C':' <b>C-class</b>.<br><br>C-class articles are substantial with a reasonably encyclopedic style, but may be missing important content, requiring substantial stylistic edits, additional sources, or compliance with one or more of Wikipedia\'s <a class="normallink" href="https://en.wikipedia.org/wiki/Wikipedia:Core_content_policies">content policies</a>.<br><br>',
    'START':' <b>start-class</b>.<br><br>Start-class articles are developing but incomplete, with a usable amount of good content but weak in many areas, including possibly un-encyclopedic prose or inadequate referencing, or very little compliance with Wikipedia\' <a class="normallink" href="https://en.wikipedia.org/wiki/Wikipedia:Core_content_policies">content policies</a>.<br><br>',
    'STUB':' a <b>stub</b>.<br><br>Stubs are either very short or very poorly written, with few sources or a rough collection of possibly irrelevant or incomprehensible information, and require significant work to become meaningful and meet Wikipedia\'s <a class="normallink" href="https://en.wikipedia.org/wiki/Wikipedia:Core_content_policies">content policies</a>.<br><br>'
}

suffixRE = re.compile(r'\w+({})$'.format('|'.join(ProductiveSuffixes)))
prefixRE = re.compile(r'({})\w+$'.format('|'.join(ProductivePrefixes)))

def getPageID(TITLE, talk=False):
    TALK = '' if not talk else 'Talk:'
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params = {
            'action' : 'query',
            'format' : 'json',
            'titles' : TALK+TITLE,
            'prop'   : 'pageprops',
        }
    ).json()

    page = next(iter(response['query']['pages'].values()))
    return page['pageid']

def getWikiCode(TITLE):
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params = {
            'action' : 'query',
            'format' : 'json',
            'titles' : TITLE,
            'prop'   : 'revisions',
            'rvprop' : 'content',
        }
    ).json()

    page = next(iter(response['query']['pages'].values()))
    return page['revisions'][0]['*']

def getPlainText(TITLE, WIKICODE):
    xmlText = '\n'.join([
        '<siteinfo>',
        '</siteinfo>',
        '<page>',
        '<title>',
        '{TITLE}',
        '</title>',
        '<id>',
        '10',
        '</id>',
        '<text>',
        '{WIKICODE}',
        '</text>',
        '</page>\n',
        ]).format(TITLE=TITLE, WIKICODE=WIKICODE)
    open('tmp.txt', 'w').write(xmlText)
    ExtractorPath = FLASK_PREFIX + 'WikiExtractor.py'
    bash.call(['bzip2', '-z', 'tmp.txt'])
    bash.call(['python3', ExtractorPath, '--no_templates', '--processes', '8', '--json', 'tmp.txt.bz2'], stdout=FNULL, stderr=bash.STDOUT)
    plaintext = ''
    with open('text/AA/wiki_00') as f:
        for line in f:
            x = line.strip('\n')
            jx = json.loads(x)
            plaintext = jx['text']

    bash.call(['rm', '-r', 'tmp.txt.bz2', 'text'])
    return plaintext

def getMediaWikiLinks(WIKICODE):
    wiki = mwparserfromhell.parse(WIKICODE)
    cls = None

    for t in wiki.filter_templates():
        if 'featured article' in t:
            cls = 'FA'
        if '-stub' in t:
            cls = 'STUB'
        if 'good article' in t:
            cls = 'GA'
    
    nWikiLinks = len(wiki.filter_wikilinks())
    nExternalLinks = len(wiki.filter_external_links())

    return nWikiLinks, nExternalLinks

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

def processPlainText(text):
    nSents   = 0
    nWords   = 0
    nAffixes = 0
    nWChars  = 0

    Sentiments = {'Doc' : SIA.polarity_scores(text)}
    Sentiments['Sen'] = {key:0. for key in Sentiments['Doc']}

    wordLengths = []
    for sent in nltk.sent_tokenize(text):

        sentimentThisSent = SIA.polarity_scores(sent)
        for key in sentimentThisSent:
            Sentiments['Sen'][key] += sentimentThisSent[key]

        nWordsThisSent   = 0
        nAffixesThisSent = 0
        nWCharsThisSent  = 0
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

    Sentiments['Sen'] = {key:safeDivide(Sentiments['Sen'][key], nSents) for key in Sentiments['Sen']}

    binnedResults = np.histogram(wordLengths, bins=10, range=(0., 20.))[0].tolist()

    sentimentList = [Sentiments['Doc'][key] for key in ('pos', 'neg', 'neu', 'compound')] + \
                    [Sentiments['Sen'][key] for key in ('pos', 'neg', 'neu', 'compound')]

    return [nSents, nWords, nSemi, nColon, nAffixes, fWordLength] + binnedResults + sentimentList

linkQuantities = ['nWikiLinks', 'nExternalLinks']

derivedQuantities = [
    'nSents',
    'nWords',
    'nSemi',
    'nColon',
    'nAffixes',
    'fWordLength'
]

binnedList = ['{:02d}_{:02d}'.format(i, i+2) for i in range(0, 20, 2)]
binnedQuantities = ['nWords_'+b for b in binnedList]

sentiListDoc = ['nDocSenti_'+s for s in ('pos', 'neg', 'neu', 'compound')]
sentiListSen = ['nSenSenti_'+s for s in ('pos', 'neg', 'neu', 'compound')]

def getFeatures(PLAINTEXT, WIKICODE):
    links  = list(getMediaWikiLinks(WIKICODE))
    values = processPlainText(PLAINTEXT)

    features = dict(zip(linkQuantities + derivedQuantities + binnedQuantities + sentiListDoc + sentiListSen, links+values))

    for key in list(features.keys()):
        if key in ('fWordLength', 'nSents') or 'Senti_' in key: continue
        features['f'+key[1:]] = safeDivide(features[key], features['nSents'])

    return features

def classify(TITLE):
    try:
        wikiCode   = getWikiCode(TITLE)
        plainText  = getPlainText(TITLE, wikiCode)
        pageID     = getPageID(TITLE)
        talkPageID = getPageID(TITLE, talk=True)
        features   = getFeatures(plainText, wikiCode)
        Class      = CLASSIFIER.classify(features)

        features.update(
            {
                'title':TITLE, 'class':Class, 'id':pageID, 'talk':talkPageID, 'color':COLORS[Class], 'flavorText':FlavorText[Class],
                'textColor':TEXTCOLORS[Class],
            }
        )
        return features
    except:
        tempDict = {'error':'ERROR', 'title':TITLE}
        return tempDict

if __name__ == '__main__':
    results = classify('Cleopatra')
    print(results['title'], results['class'])
