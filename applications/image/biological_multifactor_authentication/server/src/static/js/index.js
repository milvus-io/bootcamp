let _mediaRecorder;
let isRecording = ''; //防止两次上传
let _mediaStream;
let _chunks;

$(document).ready(function () {
    initialize();

    $("#startBtn1").click(function () {
        console.log("# 点击 startBtn");
        $("#output1").empty();
        _chunks = [];
        op = "登陆";
        _mediaRecorder.start();  //  开始录像
    });

    $("#stopBtn1").click(function () {
        console.log("# 点击 stopBtn");
        _mediaRecorder.stop(); //停止录像

    });

    $("#startBtn2").click(function () {
        console.log("# 点击 startBtn");
        $("#output1").empty();
        _chunks = [];
        op = "注册";
        _mediaRecorder.start();  //  开始录像
    });

    $("#stopBtn2").click(function () {
        console.log("# 点击 stopBtn");
        _mediaRecorder.stop(); //停止录像

    });

    $("#resetBtn").click(function () {
        console.log("# 点击 resetBtn");
        //重置
        if (isRecording !== "") {
            isRecording = "";
            _mediaRecorder.start();
        }
    });

    $("#openBtn").click(function () {
        initialize();
    });

    $("#closeBtn").click(function () {
        closeMediaStream();
    });

    $("#signup").click(function () {
        $(".middle").toggleClass("middle-flip");
    });

    $("#login").click(function () {
        $(".middle").toggleClass("middle-flip"); 
    });

}); // end $(document).ready

// 初始化摄像头设备
var initialize = function () {
    //  判断浏览器, 获得用户设备的兼容方法
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    constraints = { audio: true, video: { width: 1280, height: 720 } };

    //调用摄像头
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (mediaStream) {
            _mediaStream = mediaStream;
            console.log("# 初始化 摄像头");
            // 成功后获取视频流：mediaStream
            var video = document.getElementById('video');
            //  赋值 video 并开始播放
            video.srcObject = mediaStream;
            // video2.srcObject = mediaStream;
            video.onloadedmetadata = function (e) {
                // video2.pause();
                video.muted = true;
                video.play();
            };
            // 初始化录制器
            initMediaRecorder(mediaStream);
        });

};// end initialize

// 初始化录制器
var initMediaRecorder = function (mediaStream) {
    console.log("# 初始化 mediaRecorder");
    _chunks = [];
    // 视频格式
    let VIDEO_FORMAT = 'video/webm';
    if (!MediaRecorder.isTypeSupported(VIDEO_FORMAT)) {
        alert(format)
        alert("当前浏览器不支持该编码类型");
        return;
    }
    // 初始化 录像 mediaRecorder
    _mediaRecorder = new MediaStreamRecorder(mediaStream);
    _mediaRecorder.mimeType = VIDEO_FORMAT;
    //  当停止录像以后的回调函数
    _mediaRecorder.ondataavailable = function (data) {
        console.log("# 产生录制数据...");
        console.log(data);
        console.log("# ondataavailable, size = " + parseInt(data.size / 1024) + "KB");
        _chunks.push(data);
    };
    _mediaRecorder.onstop = function (e) {
        console.log("# 录制终止 ...");
        const fullBlob = new Blob(_chunks);
        const blobURL = window.URL.createObjectURL(fullBlob);
        console.log("blob is ?, size=" + parseInt(fullBlob.size / 1024) + "KB. "); console.log(fullBlob);
        console.log("blobURL =" + blobURL);
        // saveFile(blobURL);
        uploadFile(fullBlob);
    }
}// end initMediaRecorder

// 关闭流
var closeMediaStream = function () {
    if (!_mediaStream) return;
    console.log("# 关闭数据流");
    _mediaStream.getTracks().forEach(function (track) {
        track.stop();
    });
    _mediaStream = undefined;
    _mediaRecorder = undefined;
}

// 保存文件（产生下载的效果)
let saveFile = function (blob) {
    const link = document.createElement('a');
    link.style.display = 'none';
    link.href = blob;
    link.download = 'media_.mp4';
    document.body.appendChild(link);
    link.click();
    link.remove();
}

let uploadFile = function (blob) {
    var file = new File([blob], "media_.mp4");
    var name = document.getElementById('input-normal').value;
    console.log(name)
    var formData = new FormData();
    formData.append('file', file);
    formData.append('op', op);
    formData.append('name', name);
    console.log(formData);
    console.log("# 准备上传, fileName=" + file.name + ", size=" + parseInt(file.size / 1024) + " KB");
    var $output = $("#output1");
    if(op == "注册"){
        $output = $("#output2");
    }
    $.ajax({
        type: "POST",
        url: "/uploadvideo",
        data: formData,
        processData: false,
        contentType: false,
        success: function (result) {
            console.log(result)
            $output.prepend(result.msg);
        },
        error: function () {
            $output.prepend(op+"失败!");
        }
    });
}
