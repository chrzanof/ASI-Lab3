from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load('models/XGBoost_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = data['input']

        input_df = pd.DataFrame(input_data)

        prediction = pipeline.predict(input_df)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
