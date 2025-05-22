from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import subprocess
import time
import json
import logging
import threading
import random
from datetime import datetime
from flask import session


app = Flask(__name__)
app.secret_key = os.urandom(24)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

BASE_CHAT_FOLDER = 'chat_folders'

if not os.path.exists(BASE_CHAT_FOLDER):
    os.makedirs(BASE_CHAT_FOLDER)

def get_counter_value(chat_folder):
    # Define the path to the counter.json file
    counter_file_path = os.path.join(chat_folder.split("/")[0], 'counter.json')
    # If the counter file does not exist, create it with an initial counter of 0
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, 'w') as f:
            json.dump({'counter': 0}, f)
    # Read the current counter value from the file
    with open(counter_file_path, 'r') as f:
        counter_data = json.load(f)
        return counter_data.get('counter', 0)

def update_counter_value(chat_folder, counter_value):
    # Define the path to the counter.json file
    counter_file_path = os.path.join(chat_folder.split("/")[0], 'counter.json')
    # counter_file_path = os.path.join(chat_folder, 'counter.json')

    # Increment the counter and write it back to the counter.json file
    with open(counter_file_path, 'w') as f:
        json.dump({'counter': counter_value + 1}, f)


@app.route('/')
def website_init():
    return render_template('welcome.html')


@app.route('/negotiation')
def negotiation():
    folder_key = request.args.get('folder_key')
    return render_template('negotiation_init.html', folder_key=folder_key)

@app.route('/consent_form')
def consent_form():
    folder_key = request.args.get('folder_key')
    if folder_key:
        return render_template('consent_form.html', folder_key=folder_key)
    else:
        return "Invalid access", 400

@app.route('/attention_check')
def attention_check():
    folder_key = request.args.get('folder_key')
    return render_template('attention_check.html', folder_key=folder_key)


@app.route('/start_negotiation', methods=['POST'])
def process_negotiation():
    apples = request.form.get('apples')
    bananas = request.form.get('bananas')
    oranges = request.form.get('oranges')
    apples_importance = request.form.get('importance_apples')
    bananas_importance = request.form.get('importance_bananas')
    oranges_importance = request.form.get('importance_oranges')
    folder_key = request.form.get('folder_key')

    chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)

    # List of algorithms
    algorithms = ['stcr', 'random', 'gca']
    b_vals = [[66, 33, 33], [33, 66, 33], [33, 33, 66], [66, 66, 66], [33, 33, 33]]
    combinations = []
    for b_val in b_vals:
        for p_type in algorithms:
            combinations.append((p_type, b_val))
    # Shuffle the algorithms list
    info_dict = {}
    # Store the shuffled algorithms and the current index in the session
    info_dict['suffled_algorithms'] = algorithms
    info_dict['current_index'] = 0

    counter_val = get_counter_value('chat_folders/temp')
    algorithms = [combinations[counter_val % len(combinations)][0]]
    update_counter_value('chat_folders/temp', counter_val)
    info_dict['counter_val'] = counter_val
    with open(f'chat_folders/{folder_key}/algo_info.json', 'w') as json_file:
        json.dump(info_dict, json_file, indent=4)

    # Create subfolders for each algorithm
    for algorithm in algorithms:
        algo_folder = os.path.join(BASE_CHAT_FOLDER, f"{folder_key}/{algorithm}")
        if not os.path.exists(algo_folder):
            os.makedirs(algo_folder)

        # Store the target values for each algorithm
        target_values = {
            'apples': apples,
            'bananas': bananas,
            'oranges': oranges,
            'apples_importance': apples_importance,
            'bananas_importance': bananas_importance,
            'oranges_importance': oranges_importance,
        }
        with open(os.path.join(algo_folder, 'target_values.txt'), 'w') as f:
            json.dump(target_values, f)
        with open(os.path.join(f'chat_folders/{folder_key}', 'target_values.txt'), 'w') as f:
            json.dump(target_values, f)

    # Start with the first algorithm in the shuffled list
    algorithm = algorithms[0]

    def run_script():
        subprocess.Popen(
            ['python3', 'python_node.py', os.path.join(BASE_CHAT_FOLDER, folder_key), algorithm, str(counter_val)])

    thread = threading.Thread(target=run_script)
    thread.start()

    # Directly redirect to the chat page, bypassing the init_transition
    return redirect(url_for('chat', folder_key=folder_key, algorithm=algorithm))

@app.route('/store_consent', methods=['POST', 'GET'])
def store_consent():
    # Get consent form data from the request

    data = request.get_json()
    folder_key = data.get('folder_key')
    consent_given = data.get('consent')  # Example: consent_given is a boolean value
    print("Reached Store Consent: ", folder_key, consent_given)
    if consent_given is not None and folder_key:
        # Create the folder for storing consent data if it doesn't exist
        consent_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)
        if not os.path.exists(consent_folder):
            os.makedirs(consent_folder)

        # Get current date and time in a readable format
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Store consent in a file with the timestamp
        consent_file_path = os.path.join(consent_folder, 'consent.txt')
        with open(consent_file_path, 'a') as consent_file:
            consent_file.write(f"Consent Given: {consent_given}\n")
            consent_file.write(f"Timestamp: {timestamp}\n")
            consent_file.write("\n")  # Add a newline between entries

        return jsonify(status="Consent stored successfully"), 200

    return jsonify(status="Failed to store consent"), 400

@app.route('/target_items', methods=['GET'])
def target_items():
    folder_key = request.args.get('folder_key')
    return render_template('target_items.html', folder_key=folder_key)

@app.route('/init_transition/<folder_key>/<algorithm>', methods=['GET'])
def init_transition(folder_key, algorithm):
    # Get the target values (apples, bananas, oranges) from the request arguments
    apples = request.args.get('apples', 50)
    bananas = request.args.get('bananas', 50)
    oranges = request.args.get('oranges', 50)

    # Render the init_transition.html with the required variables
    return render_template('init_transition.html',
                           folder_key=folder_key,
                           algorithm=algorithm,
                           apples=apples,
                           bananas=bananas,
                           oranges=oranges)

@app.route('/store_prolific_id', methods=['POST'])
def store_prolific_id():
    data = request.get_json()
    prolific_id = data.get('prolific_id')

    if prolific_id:
        # Create folder_key based on timestamp
        folder_key = str(prolific_id) # str(int(time.time() * 10000))
        chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)
        i = 1

        # Check if folder exists and append _{i} if needed
        while os.path.exists(chat_folder):
            chat_folder = os.path.join(BASE_CHAT_FOLDER, f"{folder_key}_{i}")
            i += 1

        os.makedirs(chat_folder)

        # Store the Prolific ID and timestamp in the folder
        with open(os.path.join(chat_folder, 'prolific_id.txt'), 'w') as f:
            f.write(f"Prolific ID: {prolific_id}\nTimestamp: {time.time()}\n")

        return jsonify(status="ID stored", folder_key=folder_key)

    return jsonify(status="Failed to store ID"), 400


@app.route('/chat/<folder_key>/<algorithm>')
def chat(folder_key, algorithm):
    # Retrieve the target values from the previous algorithm's folder
    target_values = {}
    previous_algo_folder = os.path.join(BASE_CHAT_FOLDER, f"{folder_key}/{algorithm}")

    target_file = os.path.join(f'chat_folders/{folder_key}', 'target_values.txt')
    if os.path.exists(target_file):
        with open(target_file, 'r') as f:
            target_values = json.load(f)

    print("Target Values Check: ", target_values)

    apples = target_values.get('apples', 50)  # Default to 50 if not available
    bananas = target_values.get('bananas', 50)  # Default to 50 if not available
    oranges = target_values.get('oranges', 50)  # Default to 50 if not available

    current_apples = 50
    current_bananas = 50
    current_oranges = 50

    next_apples = 50
    next_bananas = 50
    next_oranges = 50

    return render_template('chat.html', folder_key=f"{folder_key}/{algorithm}", algorithm=algorithm,
                           apples=apples, bananas=bananas, oranges=oranges,
                           current_apples=current_apples, current_bananas=current_bananas,
                           current_oranges=current_oranges,
                           next_apples=next_apples, next_bananas=next_bananas, next_oranges=next_oranges)


@app.route('/submit_overall_satisfaction', methods=['POST'])
def submit_overall_satisfaction():
    # print("Checking Overall Satisfaction")
    data = request.get_json()
    rating = data.get('rating')

    folder_key = data.get('folder_key')
    algorithm = data.get('algorithm')
    # print("Folder_values", folder_key)
    chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key + '/' + algorithm)
    rating_file = os.path.join(chat_folder, 'satisfaction_rating.txt')
    id_file = os.path.join(chat_folder, 'prolific_id.txt')
    with open(rating_file, 'a') as f:
        f.write(f"Overall User Rating: {rating}\n")
        f.write(f"\n Ending Time: {time.time()}")
    return jsonify(status="Rating received")


@app.route('/next_scenario/<folder_key>/<algorithm>', methods=['POST', 'GET'])
def next_scenario(folder_key, algorithm):
    # Get satisfaction rating if provided
    satisfaction = request.form.get('satisfaction')
    if satisfaction:
        satisfaction_file = os.path.join(BASE_CHAT_FOLDER, folder_key, 'satisfaction_rating.txt')
        with open(satisfaction_file, 'w') as f:
            f.write(f"Overall Satisfaction Rating: {satisfaction}\n")
    return redirect(url_for('display_score', folder_key=folder_key))

@app.route('/display_score/<folder_key>/', methods=['POST', 'GET'])
def display_score(folder_key):
    try:
        # Construct the file path and load the score
        chat_folder_full = os.path.join(folder_key)
        chat_folder_full = os.path.join("chat_folders", chat_folder_full)
        score_file_path = os.path.join(chat_folder_full, "score.json")
        with open(score_file_path, 'r') as score_file:
            score_dict = json.load(score_file)

        starting_state = score_dict["Initial State"]
        final_state = [int(value) for value in score_dict["Final State"]]
        target_state = score_dict["Target State"]
        score = round(score_dict["Score"] * 100)

        # Render the score HTML page
        return render_template(
            'display_score.html',
            score=score,
            folder_key=folder_key,
            starting_state=starting_state,
            final_state=final_state,
            target_state=target_state
        )

    except Exception as e:
        return f"Error loading score: {e}", 500


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
    rating_file = os.path.join(chat_folder, 'satisfaction_rating.txt')
    counter = 0
    while not os.path.exists(offer_file):
        time.sleep(1)
        counter += 1
        if counter >= 60:
            offer = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
            with open(rating_file, 'a') as f:
                f.write(f"Offer: {offer}\n")
                f.write(f"End Time: {time.time()}\n")
            return jsonify(offer=offer, current_apples=0, current_bananas=0, current_oranges=0, next_apples=0,
                           next_bananas=0, next_oranges=0)

    if os.path.exists(offer_file):
        with open(offer_file, 'r') as f:
            offer = f.read()
        os.remove(offer_file)  # Delete the offer file after reading it

        # Store the offer in the satisfaction_rating.txt file
        with open(rating_file, 'a') as f:
            f.write(f"Offer: {offer}\n")
        if offer == "The algorithm has ended trading, please press the Stop Trading button to end the scenario":
            return jsonify(offer=offer, current_apples=0, current_bananas=0, current_oranges=0, next_apples=0,
                           next_bananas=0, next_oranges=0)
        else:
            offer = offer.replace("ST-CR", "Computer")
            offer = offer.replace("GCA", "Computer")
            offer = offer.replace("Random Agent", "Computer")
            offer = offer.replace("Computer receives", "User gives")
            # Find the start of the trade offer
            start_index = offer.index("Computer's Trade Offer:")
            # Find the start of the next section
            end_index = offer.index("Your Next State After This Trade:")

            # Extract the trade offer
            trade_offer = offer[start_index:end_index].strip()

            # Format the trade offer to include a newline
            # trade_offer = trade_offer.replace(":", ":\n")  # Ensure the title is on its own line
            trade_offer = trade_offer + "\n" + "Do you accept this offer?"

            if "Trade has Been Accepted!" in offer:
                trade_offer = "Trade has Been Accepted!" + "\n" + trade_offer
            if "Computer: I understand that you would prefer the following offer" in offer:
                start_index_2 = offer.index("Computer: I understand that you would prefer the following offer")
                end_index_2 = offer.index("Your Current State: ")
                prev_offer = offer[start_index_2:end_index_2].strip()
                trade_offer = prev_offer + "\n" + trade_offer
            lines = offer.split('\n')
            current_state_dict = {'apples': 0.0, 'bananas': 0.0, 'oranges': 0.0}
            next_state_dict = {'apples': 0.0, 'bananas': 0.0, 'oranges': 0.0}
            for line in lines:
                if f"Your Current State: " in line:
                    for item in current_state_dict.keys():
                        parts = line.split()
                        for i in range(len(parts)):
                            if item in parts[i]:
                                quantity = float(parts[i - 1])
                                current_state_dict[item] += quantity
                if "Your Next State After This Trade: " in line:
                    for item in current_state_dict.keys():
                        parts = line.split()
                        for i in range(len(parts)):
                            if item in parts[i]:
                                quantity = float(parts[i - 1])
                                next_state_dict[item] += quantity

            current_apples = current_state_dict['apples']
            current_bananas = current_state_dict['bananas']
            current_oranges = current_state_dict['oranges']

            next_apples = next_state_dict['apples']
            next_bananas = next_state_dict['bananas']
            next_oranges = next_state_dict['oranges']
            return jsonify(offer=trade_offer, current_apples=current_apples, current_bananas=current_bananas,
                           current_oranges=current_oranges, next_apples=next_apples, next_bananas=next_bananas,
                           next_oranges=next_oranges)

    return jsonify(offer="")


@app.route('/submit_satisfaction', methods=['POST'])
def submit_satisfaction():
    data = request.get_json()
    folder_key = data.get('folder_key')
    rating = data.get('rating')

    if rating and folder_key:
        chat_folder = os.path.join(BASE_CHAT_FOLDER, folder_key)
        rating_file = os.path.join(chat_folder, 'satisfaction_rating.txt')
        with open(rating_file, 'a') as f:
            f.write(f"User Rating: {rating}\n")
        return jsonify(status="Rating received")

    return jsonify(status="Failed to receive rating")

@app.route('/overall_satisfaction/<folder_key>/<algorithm>')
def overall_satisfaction(folder_key, algorithm):
    return render_template('overall_satisfaction.html', folder_key=folder_key, algorithm=algorithm)

@app.route('/trading_ended/<folder_key>', methods=['POST', 'GET'])
def trading_ended(folder_key):
    # Log the end of trading
    rating_file = os.path.join(BASE_CHAT_FOLDER, folder_key, 'prolific_id.txt')
    with open(rating_file, 'a') as f:
        f.write(f"Ending Time: {time.time()}\n")
    return redirect("https://app.prolific.com/submissions/complete?cc=C1EYUQZS")

@app.route('/trading_ended_target_fail/<folder_key>', methods=['POST', 'GET'])
def trading_ended_target_fail(folder_key):
    # Log the end of trading
    rating_file = os.path.join(BASE_CHAT_FOLDER, folder_key, 'prolific_id.txt')
    with open(rating_file, 'a') as f:
        f.write(f"Ending Time: {time.time()}\n")
        f.write(f"Failed to Enter Target Items \n")
    return redirect("https://app.prolific.com/submissions/complete?cc=CMEENM2X")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Bind to all interfaces for Docker