from flask import Flask, request, jsonify, send_from_directory
import os
import csv
import subprocess
from threading import Thread

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

# Start the server
if __name__ == '__main__':
    app.run(port=5000)
