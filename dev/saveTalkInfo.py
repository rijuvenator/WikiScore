import xml.sax
import subprocess as bash
import mwparserfromhell
import re
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('FILE')
args = parser.parse_args()

cre = re.compile(r'class\s*=\s*(stub|start|C|B|GA|A|FA|Redirect)', flags=re.I)
arc = re.compile(r'/Archive \d*')
fname = re.compile(r'current(\d{1,2})\.xml-(p.*)\.bz2')

# this code gathered from
# https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c

# defines the class
class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []

        # pageStart is for keeping track of a page open and close
        # so that ID is only filled the FIRST time, rather than for every ID tag
        self._pageStart = False

        # for 
        FNAME = 'TalkPages_{}_{}.txt'.format(*re.search(fname, os.path.basename(args.FILE)).groups())
        self._file = open(FNAME, 'w')
        print(f'Opened {FNAME} for writing')

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name == 'page':
            self._pageStart = True
        if name in ('title', 'text', 'id'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            if name == 'id':
                if self._pageStart:
                    self._values[name] = ' '.join(self._buffer)
                    self._pageStart = False
                else:
                    return
            else:
                self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            #self._pages.append((self._values['title'], self._values['text'], self._values['id']))

            tit, txt, ID = self._values['title'], self._values['text'], self._values['id']

            # this parses the txt

            # for this particular script, this will not be used
            if 'Talk:' not in tit:
                return
                wiki = mwparserfromhell.parse(txt)
                cls = None
                for t in wiki.filter_templates():
                    if 'featured article' in t:
                        cls = 'FA'
                    if '-stub' in t:
                        cls = 'STUB'
                    if 'good article' in t:
                        cls = 'GA'
                if cls is not None:
                    data.append({'ID':ID, 'Type':'A', 'Title':tit, 'Class':cls})

            # Talk: is in the title: save the classes, ID, and title
            else:
                wiki = mwparserfromhell.parse(txt)
                classList = []

                isDisamb = False
                isRedirect = False

                if re.search(arc, tit): return
                if '#REDIRECT' in txt: return
                if 'disambiguation' in tit: return
                if ':List of' in tit: return

                for t in wiki.filter_templates():

                    if 'DisambigProject' in t:
                        isDisamb = True
                        break

                    m = re.search(cre, str(t))
                    if m:
                        #print(m.group(1))
                        if m.group(1).upper() == 'REDIRECT':
                            isRedirect = True
                            break

                        classList.append(m.group(1).upper())

                if isDisamb: return
                if isRedirect: return

                classDict = {}
                for i in set(classList):
                    classDict[i] = classList.count(i)
                if len(classDict) != 0:
                    #print(classDict)
                    maxCount, maxClass = 0, ''
                    for c in classDict:
                        if classDict[c] > maxCount:
                            maxClass = c
                    #print(ID, ':::', tit.replace('Talk:', ''), ':::', ' '.join(classList))
                    self._file.write('{} ::: {} ::: {}\n'.format(ID, tit.replace('Talk:', ''), ' '.join(classList)))
                else:
                    #print(ID, ':::', tit.replace('Talk:', ''), ':::', 'None')
                    self._file.write('{} ::: {} ::: {}\n'.format(ID, tit.replace('Talk:', ''), 'None'             ))

# initializes
handler = WikiXmlHandler()
parser = xml.sax.make_parser()
parser.setContentHandler(handler)

print('XML Handler initialized; will now stream from .bz2 file')

# reads the data from a bz2 file, streaming
# then parser feeds a line, handled by the handler
# the handler initializes stuff, does NOT save stuff to pages, 

data_path = args.FILE

for line in bash.Popen(['bzcat'], 
                              stdin = open(data_path), 
                              stdout = bash.PIPE).stdout:
    parser.feed(line)
    
    # Stop when 3 articles have been found
    #if len(handler._pages) > 1000:
    #    break
