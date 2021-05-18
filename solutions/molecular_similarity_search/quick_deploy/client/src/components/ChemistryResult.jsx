import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import ZoomInIcon from "@material-ui/icons/ZoomIn";
const Result = props => {
  const { src, Molecular, Distance, style, onClick } = props;
  const useStyles = makeStyles({
    container: {
      width: "100%",
      display: "flex",
      alignItems: "start",
      marginBottom: "15px",
      ...style
    },
    structure: {
      position: "relative",
      flex: "0 0 21%",
      paddingTop: "20%",
      border: "solid 1px #60606F",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      fontSize: "50px",
      marginRight: "20px",
      cursor: "pointer"
    },
    child: {
      width: "100%",
      maxHeight: "100%",
      position: "absolute",
      top: `0`,
      bottom: `0`,
      margin: "auto",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      color: "#60606F",
      fontSize: "8vw"
    },
    label: {
      flexGrow: "1",
      height: "100%",
      display: "flex",
      justifyContent: "start",
      alignItems: "center",
      fontSize: "1vw",
      color: "#fff"
    },
    formula: {
      width: "70%",
      wordWrap: "break-word",
      wordBreak: "break-all",
      padding: "0 30px 0 0"
    },
    distance: {
      width: "30%"
    },
    icon: {
      position: "absolute",
      bottom: "10px",
      right: "10px"
    }
  });
  const classes = useStyles({});
  return (
    <div className={classes.container}>
      <div className={classes.structure}>
        <img src={src} className={classes.child} alt="" onClick={onClick} />
        <ZoomInIcon className={classes.icon} />
      </div>
      <div className={classes.label}>
        <p className={classes.formula}>{Molecular}</p>
        <p className={classes.distance}>{Distance}</p>
      </div>
    </div>
  );
};

export default Result;
