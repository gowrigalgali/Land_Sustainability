from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import csv
import subprocess
from threading import Thread
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Path to the CSV file
csv_file_path = "coordinates.csv"

# Middleware to serve static files (if needed)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.getcwd(), filename)

# Save coordinates to CSV (overwrite existing file)
@app.route('/search', methods=['POST'])
def save_coordinates():
    new_coordinates = request.json.get('coordinates')

    if not new_coordinates or len(new_coordinates) == 0:
        return jsonify({"message": "No coordinates provided"}), 400

    # Write new data to CSV file (overwrite mode)
    try:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Latitude', 'Longitude'])
            for coord in new_coordinates:
                writer.writerow([coord['lat'], coord['lng']])

        # Call the function asynchronously
        Thread(target=call_another_script, args=(csv_file_path,)).start()

        return jsonify({"message": "Coordinates saved successfully!"})

    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        return jsonify({"message": "Error saving coordinates"}), 500


def call_another_script(file_path):
    try:
        # Using subprocess to run the other Python script
        subprocess.run(['python', 'model.py', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running another script: {e}")


# Serve index.html as the main page
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

# Generate report and return the Word document
@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        # Ensure coordinates CSV exists; if the frontend sends coordinates, overwrite first
        payload = request.get_json(silent=True) or {}
        coords = payload.get('coordinates')
        if coords and isinstance(coords, list):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Latitude', 'Longitude'])
                for coord in coords:
                    writer.writerow([coord['lat'], coord['lng']])

        # Run model to generate report.docx
        result = subprocess.run(['python', 'model.py', csv_file_path], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({"message": "Failed to generate report", "stderr": result.stderr}), 500

        # Return the generated file
        report_path = os.path.join(os.getcwd(), 'report.docx')
        if not os.path.exists(report_path):
            # Fallback if model saved elsewhere
            report_path = os.path.join(os.getcwd(), 'app', 'report.docx') if os.path.exists(os.path.join(os.getcwd(), 'app', 'report.docx')) else None
        if not report_path or not os.path.exists(report_path):
            return jsonify({"message": "Report not found after generation"}), 500

        return send_file(report_path, as_attachment=True, download_name='report.docx')

    except Exception as e:
        return jsonify({"message": f"Error generating report: {str(e)}"}), 500

# Start the server
if __name__ == '__main__':
    app.run(port=5000)
