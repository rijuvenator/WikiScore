from flask import request, render_template
from flaskexample import app

import flaskexample.classifyArticle

DEFAULT = flaskexample.classifyArticle.DEFAULT

import datetime

@app.route('/')
@app.route('/index')
def index():
    STYLESHEET = render_template("styles.css", output=DEFAULT)
    return render_template("index.html", output=DEFAULT, STYLESHEET=STYLESHEET)


@app.route('/', methods=['POST'])
def results():
    title = request.form['ArticleTitle']
    if title == '':
        STYLESHEET = render_template("styles.css", output=DEFAULT)
        return render_template("index.html", output=DEFAULT, STYLESHEET=STYLESHEET)
    output = flaskexample.classifyArticle.classify(title)
    STYLESHEET = render_template("styles.css", output=output)
    return render_template("index.html", output=output, STYLESHEET=STYLESHEET)
