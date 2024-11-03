from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import subprocess
import time
import json
import platform
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

BASE_CHAT_FOLDER = 'chat_folders'

if not os.path.exists(BASE_CHAT_FOLDER):
    os.makedirs(BASE_CHAT_FOLDER)

@app.route('/')
def start():
    print("Starting")
    return render_template('consent_form.html')  # Serve the consent form on the homepage

@app.route('/consent', methods=['POST'])
def handle_consent():
    print("Checking for consent")
    consent = request.form.get('consent')
    prolific_id = request.form.get('prolific_id')  # Retrieve Prolific ID

    if consent == 'Yes':
        # You may want to store the Prolific ID somewhere or process it here
        if request.method == 'POST':
            print("Redirecting")
            return redirect(url_for('negotiation_init'))  # Redirect to negotiation initialization
        else:
            print("Rendering")
            return render_template('entry.html')
    else:
        return "You must agree to the consent form to proceed."

@app.route('/negotiation_init')
def negotiation_init():
    return render_template('negotiation_init.html')

@app.route('/negotiation', methods=['POST'])
def start_negotiation():
    apples = request.form.get('apples')
    bananas = request.form.get('bananas')
    oranges = request.form.get('oranges')
    program_type = request.form.get('program_type')
    api_key = request.form.get('api_key')  # Retrieve the API key from the form
    random_seed = request.form.get('random_seed')  # Retrieve the random seed from the form

    # Generate a folder key (for simplicity, we can use a timestamp)
    folder_key = str(int(time.time()))

    # Create chat folder
    chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)
    if not os.path.exists(chat_folder):
        os.makedirs(chat_folder)

    # Save target values
    target_values = {
        'apples': apples,
        'bananas': bananas,
        'oranges': oranges,
        'random_seed': random_seed  # Save random seed
    }
    with open(os.path.join(chat_folder, 'target_values.txt'), 'w') as f:
        json.dump(target_values, f)

    # Start the algorithm in a new terminal and pass the API key as an argument
    if platform.system() == 'Windows':
        subprocess.Popen(
            ['cmd', '/c', 'start', 'cmd', '/c', 'python python_node.py ' + program_type + ' ' + chat_folder + ' ' + api_key + ' ' + random_seed],
            shell=True)
    else:
        subprocess.Popen(['gnome-terminal', '--', 'python3', 'python_node.py', program_type, chat_folder, api_key, random_seed])

    return redirect(url_for('chat', folder_key=folder_key))

@app.route('/chat/<folder_key>')
def chat(folder_key):
    return render_template('chat.html', folder_key=folder_key)

@app.route('/send_response', methods=['POST'])
def send_response():
    data = request.get_json()
    folder_key = data.get('folder_key')
    response = data.get('response')

    if response and folder_key:
        chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)
        response_file = os.path.join(chat_folder, 'response.txt')
        with open(response_file, 'w') as f:
            f.write(response)
        return jsonify(status="Response received")

    return jsonify(status="Failed to receive response")

@app.route('/get_offer', methods=['GET'])
def get_offer():
    folder_key = request.args.get('folder_key')
    chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)
    offer_file = os.path.join(chat_folder, 'offer.txt')
    while not os.path.exists(offer_file):
        time.sleep(1)
    if os.path.exists(offer_file):
        with open(offer_file, 'r') as f:
            offer = f.read()
        os.remove(offer_file)  # Delete the offer file after reading it
        return jsonify(offer=offer)
    return jsonify(offer="")

@app.route('/trading_ended')
def trading_ended():
    return "Trading ended. Thank you for participating."

if __name__ == '__main__':
    print("Starting Website on http://127.0.0.1:5000")
    app.run(debug=True)