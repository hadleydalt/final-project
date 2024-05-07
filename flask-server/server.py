from flask import Flask, render_template

app = Flask(__name__)

@app.route("/prediction")
def predict():
    return {"members": ["Member1", "Member2", "Member3"]}

if __name__ == "__main__":
    app.run(debug=True)