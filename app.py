"""Contains all API methods"""
import os
import tracemalloc
from flask import Flask, jsonify, request
from flask_cors import CORS
from generate_text import get_aggregated_completions
from offensive_classifier import sort_offensive

app = Flask(__name__)
CORS(app)

tracemalloc.start()

@app.route('/', methods=['POST'])
def predictions_and_offensiveness():
    """Sorts predictions by offensiveness for redteaming"""
    req = request.get_json()
    generated = get_aggregated_completions(req['prompt'], req['numPredictions'])
    sorted_greedy = sort_offensive(generated['greedy'])
    sorted_beam = sort_offensive(generated['beam'])
    return_dict = {
        'attention': generated['attention'].tolist(),
        'beam': sorted_beam,
        'greedy': sorted_greedy,
        'tokens': [token.replace('Ġ', '') for token in generated['tokens']]}
    return jsonify(return_dict)

@app.route('/', methods=['GET'])
def verify_online():
    """Returns json to verify server is working"""
    return jsonify({"success": True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT'))
    app.run(debug=True, port=port)
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

