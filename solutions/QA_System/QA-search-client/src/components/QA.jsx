import React, { useState, useContext, useRef, useEffect } from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Divider, TextareaAutosize, Snackbar } from "@material-ui/core";
import SendIcon from "@material-ui/icons/Send";
import AccountCircleIcon from "@material-ui/icons/AccountCircle";
import { queryContext } from "../contexts/QueryContext";

const QA = props => {
  const textArea = useRef(null);
  const resultContainer = useRef(null);
  const [qaList, setQaList] = useState([
    { type: "question", text: "问题" },
    { type: "answer", text: "回答" }
  ]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const isMobile = props.isMobile;
  const { search } = useContext(queryContext);

  const useStyles = makeStyles({
    wrapper: {
      display: "flex",
      width: isMobile ? "auto" : "500px",
      flexDirection: "column"
    },
    content: {
      flex: isMobile ? "0 0 60vh" : "0 0 62vh",
      overflowY: "auto",
      color: "#fff",
      padding: isMobile ? "20px" : "40px",
      fontSize: isMobile ? "12px" : "15px"
    },
    textarea: {
      position: "relative",
      flex: "0 0 200px",
      padding: "20px",
      backgroundColor: "rgb(40, 41, 46)"
    },
    item: {
      display: "flex",
      marginTop: isMobile ? "20px" : " 60px"
    },
    avatar: {
      flex: "0 0 100x",
      width: "60px",
      height: "60px",
      borderRadius: "50%",
      backgroundSize: '64px',
      backgroundImage: `url(
        "data:image/jpeg;base64,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"
      )`
    },
    text: {
      position: "relative",
      display: "flex",
      marginLeft: "20px",
      alignItems: "center",
      maxWidth: "70%",
      backgroundColor: "#fff",
      padding: "10px 18px",
      color: "#000",
      borderRadius: "10px",
      ".question &": {
        backgroundColor: "#AEE5FF"
      }
    },
    send: {
      position: "absolute",
      top: isMobile ? "20px" : "50px",
      right: "30px",
      height: "60px",
      width: "60px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      backgroundColor: "#4FC4F9",
      borderRadius: "50%",
      color: "#fff",
      cursor: "pointer"
    },
    triangle: {
      position: "absolute",
      top: "16px",
      width: 0,
      height: 0,
      borderTop: "10px solid transparent",
      borderBottom: "10px solid transparent",
      ".question &": {
        borderLeft: "10px solid #AEE5FF"
      },
      ".answer &": {
        borderRight: "10px solid #fff"
      }
    }
  });
  const classes = useStyles({});

  const handleSend = e => {
    const value = textArea.current.value;
    if (!value.trim()) {
      setMessage("请输入问题");
      setOpen(true);

      setTimeout(() => {
        setOpen(false);
      }, 800);
      return;
    }
    setLoading(true);
    setQaList(list => {
      let text = textArea.current.value;
      textArea.current.value = "";
      return [...list, { type: "question", text }];
    });
    search({ query_text: value }).then(res => {
      const { status, data } = res || {};
      if (status === 200) {
        setQaList(list => {
          return [...list, { type: "answer", text: data }];
        });
        setLoading(false);
      }
    });
    if (loading) {
      return;
    }
  };

  useEffect(() => {
    resultContainer.current.scrollTop = 10000;
  }, [qaList]);

  return (
    <div className={classes.wrapper}>
      <div ref={resultContainer} className={classes.content}>
        <p>该 AI 问答系统包含33万条银行业务相关的问答。</p>
        <p> 在下方对话框中输入问题，你的金融管家小M将会给出回答。</p>
        <p style={{ color: "#B0B0B9" }}>（Demo 仅支持中文问答）</p>

        {qaList.map((v, i) => {
          if (v.type === "answer") {
            return (
              <div className={`${classes.item} answer`} key={i}>
                <div className={classes.avatar}></div>
                <div className={classes.text}>
                  <div
                    className={`${classes.triangle}`}
                    style={{ left: "-10px" }}
                  ></div>
                  <p>{v.text}</p>
                </div>
              </div>
            );
          } else {
            return (
              <div
                className={`${classes.item} question`}
                style={{ flexDirection: "row-reverse" }}
                key={i}
              >
                <AccountCircleIcon style={{ fontSize: 50 }} />
                <div className={classes.text} style={{ margin: "0 20px 0 0" }}>
                  <div
                    className={classes.triangle}
                    style={{ right: "-10px" }}
                  ></div>
                  <p>{v.text}</p>
                </div>
              </div>
            );
          }
        })}
        {loading && (
          <div style={{ textAlign: "center" }}>正在查询相关问题,请稍后...</div>
        )}
      </div>
      <Divider
        variant="middle"
        style={{ backgroundColor: "#fff", margin: " 0" }}
      />
      <div className={classes.textarea}>
        <TextareaAutosize
          ref={textArea}
          aria-label="empty textarea"
          placeholder="请输入问题，比如：银行面签后，公积金贷款多久能下来"
          rows={10}
          style={{
            width: "100%",
            boxSizing: "border-box",
            border: "none",
            backgroundColor: "#28292E",
            color: "#fff"
          }}
        />
        <div className={classes.send}>
          <SendIcon fontSize="large" onClick={handleSend}></SendIcon>
        </div>
      </div>
      <Snackbar
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
        key="top center"
        open={open}
        message={message}
      />
    </div>
  );
};

export default QA;
