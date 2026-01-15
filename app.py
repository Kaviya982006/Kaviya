from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = "secret_key_for_demo"

# Paths
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "model.pkl"
RESULTS_CSV = "static/all_model_result.csv"

# Load scaler and default model
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None
model = pickle.load(open(MODEL_PATH, "rb")) if os.path.exists(MODEL_PATH) else None

# Load all models for algorithm selection
models = {}
try:
    models = {
        "KNN": pickle.load(open("knn.pkl", "rb")),
        "Logistic Regression": pickle.load(open("log.pkl", "rb")),
        "Decision Tree": pickle.load(open("dt.pkl", "rb")),
        "Random Forest": pickle.load(open("rf.pkl", "rb")),
        "XGBoost": pickle.load(open("xgb.pkl", "rb")),
    }
except Exception:
    # If some models are missing, ignore
    pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username and password:
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for("home"))
        flash("Please enter username and password.", "error")
    return render_template("login.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    preview = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)
            preview = df.head().to_html(classes="table table-striped", index=False)
            flash("File uploaded successfully!", "success")
        else:
            flash("Please upload a CSV file.", "error")
    return render_template("upload.html", preview=preview)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST" and scaler:
        try:
            algo = request.form.get("algorithm", None)

            # Collect features safely
            profile_pic = int(request.form.get("profile_pic", 0))
            nums_username = float(request.form.get("nums_username", 0))
            fullname_words = int(request.form.get("fullname_words", 0))
            nums_fullname = float(request.form.get("nums_fullname", 0))
            name_equals_username = int(request.form.get("name_equals_username", 0))
            description_length = int(request.form.get("description_length", 0))
            external_url = int(request.form.get("external_url", 0))
            private = int(request.form.get("private", 0))
            posts = int(request.form.get("posts", 0))
            followers = int(request.form.get("followers", 0))
            follows = int(request.form.get("follows", 0))

            features = np.array([[
                profile_pic, nums_username, fullname_words, nums_fullname,
                name_equals_username, description_length, external_url,
                private, posts, followers, follows
            ]], dtype=float)

            features_scaled = scaler.transform(features)

            # Use chosen algorithm if available, else default model
            if algo and algo in models:
                pred = models[algo].predict(features_scaled)[0]
                result = f"{algo} says: {'Fake' if pred == 1 else 'Real'}"
            elif model:
                pred = model.predict(features_scaled)[0]
                result = "Fake" if pred == 1 else "Real"
            else:
                flash("No model available for prediction.", "error")

        except Exception as e:
            flash(f"Error in prediction: {e}", "error")

    return render_template("predict.html", result=result)


@app.route("/performance")
def performance():
    df = None
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
    return render_template(
        "performance.html",
        table=df.to_html(index=False, classes="table") if df is not None else None
    )


@app.route("/chart")
def chart():
    return render_template("chart.html")
from flask import jsonify
import os

@app.route("/debug")
def debug():
    debug_info = {}

    # Check models
    model_files = ["dt.pkl", "knn.pkl", "log.pkl", "model.pkl", "rf.pkl", "scaler.pkl", "xgb.pkl"]
    debug_info["models_found"] = [f for f in model_files if os.path.exists(f)]

    # Check templates
    debug_info["templates_found"] = os.listdir("templates") if os.path.exists("templates") else []

    # Check static files
    debug_info["static_found"] = os.listdir("static") if os.path.exists("static") else []

    # Python + package versions
    try:
        import pandas, sklearn, numpy
        debug_info["versions"] = {
            "python": os.sys.version,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__
        }
    except Exception as e:
        debug_info["versions_error"] = str(e)

    return jsonify(debug_info)



if __name__ == "__main__":
    app.run(debug=True)
