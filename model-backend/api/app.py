from flask import Flask, jsonify, request
import sys
import os

# Add the parent directory of "model2" to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predict import predict_keywords_from_abstract

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the backend server!"

@app.route('/api/extract', methods=['POST'])
def post_data():
    # Get the abstract text from the request
    abstract = request.get_json().get('abstract')
    
    # Predict keywords from the abstract
    keywords = predict_keywords_from_abstract(abstract, min_keywords=5)
    
    # Return the keywords as a JSON response
    return jsonify(keywords)

if __name__ == '__main__':
    app.run(debug=True)