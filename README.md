# WikiScore

## Overview

The _WikiScore_ Wikipedia Article Classifier can be found at [wikiscore.xyz](http://wikiscore.xyz).

It classifies Wikipedia articles, given a title, according to Wikipedia's own content assessment scheme, ranging from featured article to stub.

It may be used to classify an unassessed article, as an indication of what sort of editing efforts may best serve the article and the community.

This repository consists of `dev/`, where the scripts used to develop the classifier are stored, and `flaskexample/`, the directory for the Flask app that powers the website (deployed on AWS).

## Installation

This repository is written in Python 3.7.4.

### WikiScore Flask App
The Flask app requires the following three packages:

  * `Flask==1.1.1`
  * `mwparserfromhell==0.5.4`
  * `nltk==3.4.5`

NLTK requires the following submodules:

  * `nltk.download('punkt')`
  * `nltk.download('vader_lexicon')`
  * `nltk.download('stopwords')`

### WikiScore `dev` Code
Parsing, organizing, and training requires these packages:

  * `wikipedia==1.4.0`
  * `numpy==1.16.4`
  * `matplotlib==3.1.1`
  * `scikit-learn==0.21.2`
  * `pandas==0.25.1`

### Installation
Install them with

```bash
# Install the packages for the flask app
pip3 install Flask==1.1.1            \
             mwparserfromhell==0.5.4 \
             nltk==3.4.5             \

# Install the NLTK submodules
python3 <(echo "import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
")

# Install the packages for the analysis
pip3 install wikipedia==1.4.0     \
             numpy==1.16.4        \
             matplotlib==3.1.1    \
             scikit-learn==0.21.2 \
             pandas==0.25.1
```

## Running the Flask app

With the installation above, everything should work.

Run `run.py` from the repository folder and navigate to `localhost:5000`

`classifyArticle.py` should work standalone, changing article title as appropriate


## Reproducing the model

```bash
### Getting Wikipedia
cd dev/
DEVDIR=$(pwd)

mkdir -p Data/Wikipedia/
# wget the dump files

### Extracting Assessment Labels
python3 saveTalkInfo.py Data/Wikipedia/FILE.bz2
python3 turnDumpIntoJson.py
mkdir JSONTalks/
mv *.json JSONTalks/
mkdir TalkTxt/
mv Talk*.txt TalkTxt/

### Extracting Plaintext
git clone https://github.com/attardi/wikiextractor.git
mkdir WikiExtractor/data/
cd WikiExtractor/wikiextractor/
python3 WikiExtractor.py -o ../data --no_templates --processes 8 --json \
        ${DEVDIR}/Data/Wikipedia/FILE.bz2
cd ${DEVDIR}/WikiExtractor/data/
cat text/*/wiki_* > plaintext.json

### Saving the Main Dataframe
cd $DEVDIR
python3 pipeline.py Data/Wikipedia/FILE.bz2 WikiExtractor/data/plaintext.json

# change nDocuments and read_pickle in this file as required
python3 trainModel.py
```

### Getting Wikipedia

  * Go to `dev/`
  * Download the 30 GB of `enwiki-20190901-pages-meta-current` Wikipedia XML dump files as compressed `.bz2` from [here](https://dumps.wikimedia.org/enwiki/20190901/) and put them in `Data/Wikipedia/`

### Extracting Assessment Labels
  * Run `saveTalkInfo.py` with `Data/Wikipedia/*.bz2` arguments, one by one, to get talk pages with ID, title, and assessment classes. This will write several .txt files to the current directory
  * Run `turnDumpIntoJson.py` in that directory to make several `.json` files, and put them into a new folder `JSONTalks/`

### Extracting Plaintext
  * Clone [WikiExtractor](https://github.com/attardi/wikiextractor) into this directory
  * Make directory `WikiExtractor/data/`, go to `WikiExtractor/wikiextractor/`, and run

```bash
python3 WikiExtractor.py -o ../data --no_templates --processes 8 --json FILE.bz2
```

where `FILE.bz2` is the `.bz2` files in Data/Wikipedia (so probably `../../Data/Wikipedia/FILE.bz2`)

  * Go to `WikiExtractor/data/` and cat all of the JSON files into one; I called it `plaintext.json`

### Saving the Main Dataframe
  * Go back to `dev/`
  * Now we have all of the ingredients for `pipeline.py`:

	  * `DUMP_PATH` should be `Data/Wikipedia/FILE.bz2`
	  * `PLAINTEXT_PATH` should be `WikiExtractor/data/plaintext.json`
	  * `JSONTalks/` should be in the current directory
  * Run `pipeline.py`. After a very long time, you should have a pickled `pandas` dataframe called `testPickle.pkl`. I renamed this to `BigPickle.pkl`
  * Now `trainModel.py` can be run, changing the `read_pickle` to whatever the `.pkl` files are, and tweaking `nDocuments` to small or large values for testing purposes

The output is `model.pkl`, which goes in the `static/` folder of the Flask app.
