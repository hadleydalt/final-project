from flask import Flask, render_template
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

template_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
template_dir = os.path.join(template_dir, 'frontend')
template_dir = os.path.join(template_dir, 'public')

app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = 'shhh'

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

@app.route("/prediction", methods=['GET',"POST"])
def predict():
    form = UploadFileForm()
    return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)