from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)


def get_external_data(provider_id, category):
    api_url = f"http://localhost:5000/providers/api/{provider_id}/{category}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f"Failed to fetch data. Status code: {response.status_code}"}


def generate_recommendation(providers_data):
    # print("Providers Data:", providers_data)  # Add this line for debugging

    # Sort providers based on criteria
    sorted_providers = sorted(
        providers_data,
        key=lambda x: (
            -x['user_hireCount'],
            -x['user_rating'],
            x['user_respondTime']
        )
    )

    # print("Sorted Providers:", sorted_providers)  # Add this line for debugging

    # Return _id of the top provider
    return sorted_providers[0]['_id']


@app.route('/api/v1/hello', methods=['GET'])
def hello():
    provider_id = request.args.get('id')
    category = request.args.get('category')

    external_data = get_external_data(provider_id, category)
    recommendation = generate_recommendation(external_data)

    return jsonify({'recommendation': recommendation})
    # return jsonify({'message': 'Hello, World! 2', 'recommendation': external_data})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
