import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import Logo from "../containers/Logo.svg";

const Header = props => {
  const useStyles = makeStyles({
    header: {
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      height: "60px",
      display: "flex",
      alignItems: "center",
      paddingLeft: "40px",
      color: "#fff",
      backgroundColor: "#101010",
      boxShadow: "0 12px 24px 0 rgba(0,0,0,0.25)"
    },
    text: {
      paddingTop: "6px",
      marginLeft: "40px",
      lineHeight: "22px"
    }
  });
  const classes = useStyles({});
  return (
    <header className={classes.header}>
      <img src={Logo} width="150px" alt="logo" />
      <p className={classes.text}>AI 问答系统</p>
    </header>
  );
};

export default Header;
