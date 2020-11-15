# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_render_template]

import googleapiclient
from googleapiclient import discovery
from io import open
import numpy as np
#from google.cloud import storage
import json
#from tensorflow.keras.preprocessing.text import Tokenizer

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

word2seq = {'UNK': 1,
     'a': 2,
     'e': 3,
     'n': 4,
     'i': 5,
     'r': 6,
     'l': 7,
     's': 8,
     'h': 9,
     'y': 10,
     'o': 11,
     't': 12,
     'd': 13,
     'm': 14,
     'k': 15,
     'c': 16,
     'u': 17,
     'j': 18,
     'b': 19,
     'v': 20,
     'z': 21,
     'g': 22,
     'w': 23,
     'p': 24,
     'f': 25,
     'q': 26,
     'x': 27}

seq2word = {v: k for k, v in word2seq.items()}

ml = discovery.build('ml', 'v1')

@app.route('/', methods=['GET', 'POST'])
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    names = []
    if request.method == 'POST':
        name1 = ''
        # strip invalid chars
        for c in request.form['name'].lower(): 
            if c in word2seq.keys():
                name1 += c
        # store as names[0]
        names.append(name1)
        if len(names[0]) > 0:
            names = get_names(names)

    return render_template('index.html', names=[s.capitalize() for s in names])

def get_names(names):
    temperature = 0.25
    max_len = 12
    randomness = True
    seed_text = names[-1]

    # convert to sequence
    input_eval = [word2seq[c] for c in seed_text.lower()]
    # pad to 14 items - model requires this
    input_eval = [0] * (14-len(input_eval)) + input_eval
    
    request_body = { 'instances': [input_eval] }
    request = ml.projects().predict(
        name='projects/sys6016codeathon3/models/namegen/versions/v1',
        body=request_body)

    response = request.execute()
    if 'error' in response:
        # break
        print('error: ' + str(response))
        return names
    
    predictions = np.array(response['predictions'][0]['dense_3'])
    # remove the batch dimension
    #predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    #if randomness:
    #    predictions = predictions / temperature
    #    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    #else:
    predicted_id = np.argmax(predictions)

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state

    if predicted_id > 1 and len(seed_text) < max_len:
        seed_text += str(seq2word[predicted_id])
        names.append(seed_text)
        names = get_names(names)

    return names

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8085, debug=True)
# [END gae_python38_render_template]
