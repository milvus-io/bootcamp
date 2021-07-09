import React from "react";
import { makeStyles, Theme } from "@material-ui/core";

const useStyles = makeStyles((theme: Theme) => ({
  root: {
    background: "#000",
    color: "#fff",
    padding: "20px 24px",
    textAlign: "left",
  },
  headerTitle: {
    fontSize: "20px",
    lineHeight: "24px",
    display: "inline-block",
    fontWeight: "bold",
  },
  subTitle: {
    fontSize: "12px",
    lineHeight: "15px",
    marginLeft: "16px",
    letterSpacing: "0.1em",
    display: "inline-block",
    textTransform: "uppercase",
  },
}));

const Header = () => {
  const classes = useStyles();
  return (
    <header className={classes.root}>
      <span className={classes.headerTitle}>Audio Search</span>
      <span className={classes.subTitle}>POWERED BY MILVUS</span>
    </header>
  );
};

export default Header;
