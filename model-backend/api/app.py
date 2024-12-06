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
    num_keywords = request.get_json().get('num_keywords')
    
    num_keywords = num_keywords if num_keywords else 5
    
    # Predict keywords from the abstract
    keywords = predict_keywords_from_abstract(abstract, min_keywords=num_keywords)
    
    # Return the keywords as a JSON response
    return jsonify(keywords)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)