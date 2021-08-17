import os
import sys
import pathlib

from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename

# base.pyのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + '/../' )
from executor.infer import infer_img


def is_alowed_file(filename, allowed_extension):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

app = Flask(__name__)

UPLOAD_FOLDER = './app/static/uploads'
ALLOWED_EXTENSIONS = ['jpg', 'png']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        input_img_file = request.files['input_img_file']
        if input_img_file and is_alowed_file(input_img_file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(input_img_file.filename)
            input_img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_img_url = '../static/uploads/' + filename
            # 推論
            output_img_file = infer_img(UPLOAD_FOLDER + '/' + filename)
            # output_{filename}で保存
            output_img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename))
            output_img_url = '../static/uploads/output_' + filename

            return render_template('index.html', input_img_url=input_img_url, output_img_url=output_img_url)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)