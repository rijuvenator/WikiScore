import xml.sax
import mwparserfromhell

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

import argparse

logging.basicConfig(format='[%(asctime)s]: %(message)s', level=logging.DEBUG)

###################
#### CONSTANTS ####
###################

parser = argparse.ArgumentParser()
parser.add_argument('DUMPPATH' , default='')
parser.add_argument('PLAINTEXT', default='')
args = parser.parse_args()

DUMP_PATH      = args.DUMPPATH
PLAINTEXT_PATH = args.PLAINTEXT

#DUMP_PATH      = 'Data/Wikipedia/enwiki-20190901-pages-meta-current1.xml-p10p30303.bz2'
#PLAINTEXT_PATH = 'WikiExtractor/data/plaintext_1.json'

cre = re.compile(r'class\s*=\s*(stub|start|C|B|GA|A|FA|Redirect)', flags=re.I)
arc = re.compile(r'/Archive \d*')

##################################
#### XML PARSER AND MW PARSER ####
##################################

# this code gathered from
# https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c

# defines the XML handler class
class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None

        self._page = None

        self._justEnded = False
        self._noIDYet = False

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name == 'page':
            self._justEnded = False
            self._noIDYet = True
        if name in ('title', 'text', 'id'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            if name == 'id':
                if self._noIDYet:
                    self._values[name] = ' '.join(self._buffer)
                    self._noIDYet = False
                else:
                    return
            else:
                self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            if 'Talk:' in self._values['title']: return
            #self._pages.append((self._values['title'], self._values['text'], self._values['id']))
            #self._page = (self._values['title'], self._values['text'], self._values['id'])
            self._page = (self._values['title'], self._values['text'], self._values['id'])
            self._justEnded = True

# generator function for pages from the stream
# this will be input for the MWParser so that I don't have to run over the documents multiple times
def getPagesFromDump(DUMPFILE):
    handler = WikiXmlHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    for line in bash.Popen(['bzcat'], stdin = open(DUMPFILE), stdout = bash.PIPE).stdout:
        parser.feed(line)
        
        if handler._justEnded:
            yield handler._page

# parses pages from the stream
# now the MediaWiki text isn't actually in memory
def parseXMLPage(page):
    tit, txt, ID = page

    wiki = mwparserfromhell.parse(txt)
    
    cls = None
    
    # get any class information that's already on the page
    for t in wiki.filter_templates():
        if 'featured article' in t:
            cls = 'FA'
        if '-stub' in t:
            cls = 'STUB'
        if 'good article' in t:
            cls = 'GA'
    
    nWikiLinks = len(wiki.filter_wikilinks())
    nExternalLinks = len(wiki.filter_external_links())
    
    return {'ID':ID, 'Title':tit, 'Class':cls, 'nWikiLinks':nWikiLinks, 'nExternalLinks':nExternalLinks}

# given a list of classes, figure out the one that occurs most often
# and take it as the class for this article. suitable for .apply
def decideClass(x):
    c = max(set(x), key = x.count)
    if c != 'None':
        return c
    return None


##############
#### RUN! ####
##############

if __name__ == '__main__':

    DEBUG = False
    LOG   = True

    # get dataframe from XML: media wiki links, title, ID, etc.
    # uses the XML parser, the generator function, and the parser function
    # final result: DF with just metadata
    DF = pd.DataFrame()
    count = 0
    for page in getPagesFromDump(DUMP_PATH):
        temp = pd.DataFrame({key:[val] for key, val in parseXMLPage(page).items()})
        DF = DF.append(temp)

        count += 1
        if count%1000 == 0:
            logging.info(f'Done {count}')

        if DEBUG:
            if count == 1000:
                break

    if LOG: logging.info('Got DF')

    # WikiExtractor: python3 WikiExtractor.py -o ../data --no_templates --processes 8 --json FILE.bz2
    # get plaintext with ID and URL from WikiExtractor
    # loads from JSON
    # final result: plaintextDF with metadata and text
    plaintextDF = pd.DataFrame()
    with open(PLAINTEXT_PATH) as f:
        count = 0
        for line in f:
            x = line.strip('\n')
            jx = json.loads(x)
            try:
                jx['text'] = jx['text'].split('\n\n', 1)[1]
            except:
                pass
            
            temp = pd.DataFrame({key:[val] for key, val in jx.items()})
            plaintextDF = plaintextDF.append(temp)

            count += 1
            if count%1000 == 0:
                logging.info(f'Done {count}')

            if DEBUG:
                if count == 1000:
                    break

    # rename columns
    plaintextDF = plaintextDF.rename(columns={'id':'ID', 'text':'Text', 'url':'URL'})

    # make sure ID is int
    plaintextDF = plaintextDF.astype({'ID':'int64'})
    DF = DF.astype({'ID':'int64'})

    if LOG: logging.info('Got plaintext')

    # merge number 1: plaintext + metadata DF
    mergedDF_1 = pd.merge(DF, plaintextDF, on='ID')

    if LOG: logging.info('Got first merge')

    # talk pages: I have extract all into files, then processed into JSON
    # final result 
    talkDF = pd.DataFrame()
    for fname in glob.glob('JSONTalks/*'):
        temp = pd.DataFrame(json.load(open(fname)))
        talkDF = talkDF.append(temp)

    # rename columns
    talkDF = talkDF.rename(mapper={0:'ID', 1:'Title', 2:'Classes'}, axis=1)

    # this is the actual labeled category for each article
    talkDF['Class_talk'] = talkDF['Classes'].apply(decideClass)

    if LOG: logging.info('Got talk')

    # merge number 2: merge1 + talk pages categories
    mergedDF_2 = pd.merge(mergedDF_1, talkDF, on='Title', suffixes=('_article', '_talk'))

    if LOG: logging.info('Got second merge')

    # pickle!
    mergedDF_2.to_pickle('testPickle.pkl')

    if LOG: logging.info('Pickled')
