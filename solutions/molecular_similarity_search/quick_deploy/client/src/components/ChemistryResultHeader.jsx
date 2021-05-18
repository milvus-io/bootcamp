import React from "react";
import { makeStyles } from "@material-ui/core/styles";

const ResultHeader = props => {
  const { title, Molecular, Distance, style } = props;
  const useStyles = makeStyles({
    container: {
      width: "100%",
      display: "flex",
      alignItems: "start",
      ...style
    },
    structure: {
      position: "relative",
      flex: "0 0 21%",
      paddingTop: "4%",
      display: "flex",
      justifyContent: "start",
      alignItems: "center",
      marginRight: "20px"
    },
    child: {
      width: "100%",
      maxHeight: "100%",
      position: "absolute",
      top: `0`,
      bottom: `0`,
      margin: "auto",
      display: "flex",
      justifyContent: "start",
      paddingLeft:'10px',
      alignItems: "center",
      color: "#fff",
      fontSize: "1vw",
    },
    label: {
      flexGrow: "1",
      height: "100%",
      display: "flex",
      justifyContent: "start",
      alignItems: "center",
      fontSize: "1vw",
      color: "#fff",
    },
    formula: {
      width: "70%",
      textAlign: "left"
    },
    distance: {
      width: "30%",
      textAlign: "left"

    }
  });
  const classes = useStyles({});
  return (
    <div className={classes.container}>
      <div className={classes.structure}>
        <p className={classes.child}>{title}</p>
      </div>
      <div className={classes.label}>
        <p className={classes.formula}>{Molecular}</p>
        <p className={classes.distance}>{Distance}</p>
      </div>
    </div>
  );
};

export default ResultHeader;
