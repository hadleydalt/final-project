from flask import Flask, render_template
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from predictor import calc_prediction

app = Flask(__name__) 
app.config['SECRET_KEY'] = 'shhh'
app.config['UPLOAD_FOLDER'] = 'static/files'

l1 = 0.0
l2 = 0.25

class UploadFileForm(FlaskForm):
    file = FileField("Upload a Video")
    submit = SubmitField("Upload File")

@app.route("/", methods=['GET',"POST"])
def predict():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        #file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        file_path = ""
        to_print = None
        if len(file.filename) > 3:
            suffix = file.filename[len(file.filename) - 3:].lower()
            if suffix == 'mp4' or suffix == 'mov':
                file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
                file_path = "files" + os.sep + file.filename
                result = generate_prediction(file_path)
                blinks = result['blinks']
                print("blinks is")
                print(blinks)
                time = result['time']
                #bpm = (blinks * 60.0) / time
                bps = blinks / time
                print("bps is")
                print(bps)
                if bps <= l1:
                    level = "1"
                elif bps <= l2:
                    level = "2"
                else:
                    level = "3"
                #to_print = result['hello']
        return render_template('index.html', form=form, file_path=file_path, level=level, blinks=blinks, time=time) 
    return render_template('index.html', form=form, file_path=None)

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/generate", methods=['GET',"POST"])
def generate_prediction(path):
    '''Disabling calc_prediction call FOR NOW because it was confusing the program, since it's not fully built yet!!'''
    #return {"blinks":3, "time":3.35}
    counter, time = calc_prediction(path)
    return {"blinks":counter, "time":time}

if __name__ == "__main__":
    app.run(debug=True)