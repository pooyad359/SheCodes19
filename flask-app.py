from flask import Flask, render_template, redirect, flash, request, url_for
import os
from matplotlib.pyplot import imread
from fastai.vision import load_learner, open_image

ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif'])
app=Flask(__name__)
app.secret_key = "secret key"
path=os.path.abspath(os.curdir)

learn = load_learner(path,'model/model.pkl')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit():
    print(request.url)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file=request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        img_path=(os.path.join(path,'static',filename))
        file.save(img_path)
        print(img_path)
        img=open_image(img_path)
        pred_class,_,pred_conf = learn.predict(img)
        conf=pred_conf.max().numpy()
        output_string=f'{pred_class.__str__().title()} [ {conf*100:.4} % ]'
        return render_template('prediction.html',image=filename, prediction=output_string)
    else:
        flash('The file type is not allowed')
        return redirect(request.url)


    
if __name__=='__main__':
    print('Server starting')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
