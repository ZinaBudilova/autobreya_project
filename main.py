from flask import Flask, render_template, request
import re
from Classic import *
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form["query"]
    pattern=re.compile('^(([А-Яа-я]+|"[А-Яа-я]+"|[A-Z]{1,5})(\ |\+))*([А-Яа-я]+|"[А-Яа-я]+"|[A-Z]{1,5})$')
    if pattern.fullmatch(query):
        json_result = a.process_query(query)
        if type(json_result) is dict:
            error = "Данных по вашему запросу не найдено."
            return render_template('error.html', error=error)
        result = json.loads(json_result)
        return render_template('index.html', query=query, result=result)
    else:
        error = "Ошибка! Неправильный формат запроса."
        return render_template('error.html', error=error)

if __name__ == '__main__':
    with open("corpus", "rb") as f:
        a = pickle.load(f)
    app.run(debug=False)