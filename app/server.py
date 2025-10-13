from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import csv
import subprocess

app = Flask(__name__)
CORS(app)

CSV_PATH = os.path.join(os.getcwd(), 'coordinates.csv')
LAKE_CSV = os.path.join(os.getcwd(), 'output_lake.csv')

@app.route('/search', methods=['POST'])
def save_coordinates():
    try:
        payload = request.get_json(silent=True) or {}
        coords = payload.get('coordinates')
        if not coords or not isinstance(coords, list):
            return jsonify({"message": "No coordinates provided"}), 400

        with open(CSV_PATH, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Latitude', 'Longitude'])
            for c in coords:
                w.writerow([c['lat'], c['lng']])

        return jsonify({"message": "Coordinates saved successfully!"})
    except Exception as e:
        return jsonify({"message": f"Error saving coordinates: {str(e)}"}), 500

@app.route('/generate-enhanced-report', methods=['POST'])
def generate_enhanced_report():
    try:
        payload = request.get_json(silent=True) or {}
        coords = payload.get('coordinates')
        if coords and isinstance(coords, list):
            with open(CSV_PATH, mode='w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Latitude', 'Longitude'])
                for c in coords:
                    w.writerow([c['lat'], c['lng']])

        cmd = ['python', 'enhanced_model_fast.py', CSV_PATH, LAKE_CSV, '--fast']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
        if result.returncode != 0:
            return jsonify({"message": "Failed to generate enhanced report", "stderr": result.stderr}), 500

        report_path = os.path.join(os.getcwd(), 'report.docx')
        if not os.path.exists(report_path):
            return jsonify({"message": "Enhanced report not found after generation"}), 500

        return send_file(report_path, as_attachment=True, download_name='enhanced_report.docx')

    except subprocess.TimeoutExpired:
        return jsonify({"message": "Enhanced report generation timed out. Please try again."}), 500
    except Exception as e:
        return jsonify({"message": f"Error generating enhanced report: {str(e)}"}), 500

@app.route('/')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(port=5001)


