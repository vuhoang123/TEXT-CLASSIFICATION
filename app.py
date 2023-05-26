from flask import Flask, request, render_template
from main_CNN import predict_category
import json
app = Flask(__name__, template_folder='template')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    text = request.form['text']
    return json.dumps({'prediction': predict_category(text)})


if __name__ == '__main__':
    app.run(debug=True)
