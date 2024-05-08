from flask import Flask, render_template
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

app = Flask(__name__) 
app.config['SECRET_KEY'] = 'shhh'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

@app.route("/prediction", methods=['GET',"POST"])
def generate():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        file_path = ""
        to_print = None
        if len(file.filename) > 3:
            suffix = file.filename[len(file.filename) - 3:]
            if suffix == 'mp4':
                file_path = "files/" + file.filename
                result = fetchtest()
                to_print = result['hello']
        return render_template('index.html', form=form, result=to_print, file_path=file_path) 
    return render_template('index.html', form=form, file_path=None)

@app.route("/fetchtest", methods=['GET',"POST"])
def fetchtest():
    return {"hello":"these are the results!!!"}

if __name__ == "__main__":
    app.run(debug=True)