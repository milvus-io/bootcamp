import React from "react";
import { makeStyles } from "@material-ui/core/styles";

const Header = props => {
  const useStyles = makeStyles({
    header: {
      display: "flex",
      alignItems: "center",
      padding: "24px",
      color: "#000",
      backgroundColor: "#fff",
      borderBottom: "1px solid #B0B0B9"
      // boxShadow: "0 12px 24px 0 rgba(0,0,0,0.25)"
    },
    text: {
      marginRight: "20px"
    }
  });
  const classes = useStyles({});
  return (
    <header className={classes.header}>
      <p className={classes.text}>AI 问答系统 powered by Milvus</p>
      {props.loading && <p>思考中。。。</p>}
    </header>
  );
};

export default Header;
