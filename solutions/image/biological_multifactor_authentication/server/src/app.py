from flask import Flask, render_template, request,Response,jsonify
import json
import cv2
from authentication_milvus import *
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/video_sample/')
def video_sample():

    return render_template('camera.html')


@app.route('/uploadvideo', methods=["POST"])
def uploadvideo():
    if request.method == "POST":
        file = request.files.get("file")
        op = request.form["op"]
        name = request.form["name"]
        file_name = file.filename
        file.save(file_name)
        print(request.files)
        if(op=="登陆"):
            create_collection()
            flag, name = search_collection()
            if flag:
                jsondata = json.dumps({'name': name, 'msg': name+' 登陆成功'})
            else:
                jsondata = json.dumps({'name': name, 'msg': '登陆失败'})
            result = Response(response=jsondata, content_type='application/json')
            return result
        else:
            create_collection()
            insert_embedding(name)
            jsondata = json.dumps({'name': name, 'msg': name+' 注册成功'})
            result = Response(response=jsondata, content_type='application/json')
            return result

WEB_PORT = os.getenv('WEB_PORT',default=5000)

if __name__ == '__main__':

    app.run(debug=True,host='0.0.0.0',port=WEB_PORT)