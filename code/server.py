from flask import Flask, render_template
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

app = Flask(__name__) 
app.config['SECRET_KEY'] = 'shhh'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("Upload a Video")
    submit = SubmitField("Upload File")

@app.route("/", methods=['GET',"POST"])
def predict():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        file_path = ""
        to_print = None
        if len(file.filename) > 3:
            suffix = file.filename[len(file.filename) - 3:].lower()
            if suffix == 'mp4' or suffix == 'mov':
                file_path = "files/" + file.filename
                result = generate_prediction()
                to_print = result['hello']
        return render_template('index.html', form=form, result=to_print, file_path=file_path) 
    return render_template('index.html', form=form, file_path=None)

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/generate", methods=['GET',"POST"])
def generate_prediction():
    return {"hello":"78%"}

if __name__ == "__main__":
    app.run(debug=True)