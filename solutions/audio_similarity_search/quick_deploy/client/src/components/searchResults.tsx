import React, { useState } from "react";
import { Theme, makeStyles } from "@material-ui/core/styles";
import AudioPlayer from "./audioPlayer";

const useStyles = makeStyles((theme: Theme) => ({
  resultWrapper: {
    height: "100%",
    width: "100%",
  },
  tableHeader: {
    width: "100%",
    padding: "16px 0",
    display: "flex",
    lineHeight: "24px",
  },
  section1: {
    width: "20px",
    marginRight: "24px",
  },
  section3: {
    display: "flex",
    flexGrow: 1,
    alignItems: "center",
    marginLeft: "72px",
  },
  section4: {
    marginRight: "24px",
  },
  section5: {
    display: "flex",
    alignItems: "center",
    marginRight: "24px",
  },
  tableBody: {
    width: "100%",
    height: "calc(100% - 24px - 32px)",
    background: "#fff",
    overflowY: "auto",
  },
  noData: {
    textAlign: "center",
    padding: "16px 0",
    lineHeight: "24px",
  },
}));

type PropsType = {
  rows: {
    name: string;
    distance: number;
    audioSrc: string;
  }[];
  classname?: string;
};

const Search: React.FC<PropsType> = ({ rows, classname }) => {
  const classes = useStyles();

  return (
    <div className={`${classes.resultWrapper} ${classname}`}>
      <div className={classes.tableHeader}>
        <span className={classes.section1}>#</span>
        <span className={classes.section3}>Name</span>
        <span className={classes.section5}>Distance</span>
      </div>
      <div className={classes.tableBody}>
        {rows.length ? (
          rows.map((row, index) => (
            <AudioPlayer {...row} index={index + 1} key={row.name} />
          ))
        ) : (
          <p className={classes.noData}>No Data</p>
        )}
      </div>
    </div>
  );
};
export default Search;
