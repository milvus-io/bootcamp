import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import { texts } from "../lan";
const lanKey = window._env_ && window._env_.LAN === "cn" ? "cn" : "en";
console.log(window._env_);
const { title } = texts[lanKey];
const Header = (props) => {
  const useStyles = makeStyles({
    header: {
      display: "flex",
      alignItems: "center",
      padding: "24px",
      color: "#000",
      backgroundColor: "#fff",
      borderBottom: "1px solid #B0B0B9",
      // boxShadow: "0 12px 24px 0 rgba(0,0,0,0.25)"
    },
    text: {
      marginRight: "20px",
    },
  });
  const classes = useStyles({});
  return (
    <header className={classes.header}>
      <p className={classes.text}>{title}</p>
    </header>
  );
};

export default Header;
