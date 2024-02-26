import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import pickle
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the trained KNeighborsClassifier model
model = pickle.load(open("model.pickle", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        output = "Setosa"
    elif prediction[0] == 1:
        output = "Versicolor"
    else:
        output = "Virginica"

    return redirect(url_for("results", prediction_text=output))


@app.route("/results/<prediction_text>")
def results(prediction_text):
    return render_template("results.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
