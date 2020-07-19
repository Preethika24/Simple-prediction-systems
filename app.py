import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
model = load_model("insurance.h5")
import tensorflow as tf
global graph
graph = tf.get_default_graph()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    a = request.form["age"]
    b = request.form["sex"]
    c = request.form["bmi"]
    d = request.form["children"]
    e = request.form["smoker"]
    f = request.form["region"]
    g = request.form["bill"]
    h = request.form["alcohol"]
    total = [[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h)]]
    with graph.as_default():
     pred = model.predict(np.array(total))
    return render_template('index.html', y="Predicted charges is"  +str(pred[0][0]))



if __name__ == "__main__":
    app.run(debug=True)