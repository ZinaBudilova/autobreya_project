from flask import Flask, render_template, request
# from ... import Searcher

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form["query"]
    #result = Searcher.process_query(query)
    return render_template('index.html', query=query) #result=result

if __name__ == '__main__':
    app.run(debug=False)