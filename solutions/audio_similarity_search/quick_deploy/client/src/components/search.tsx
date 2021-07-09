import React, { useState, useRef } from "react";
import { Theme, makeStyles } from "@material-ui/core/styles";
import SearchResults from "./searchResults";
import AudioPlayer from "./audioPlayer";
import Button from "@material-ui/core/Button";

const useStyles = makeStyles((theme: Theme) => ({
  searchWrapper: {
    height: "100%",
  },
  title: {},
  searchButton: {
    marginBottom: "20px",
  },
  uploadWrapper: {},
  tableWrapper: {},
  targetFileWrapper: {},
  button: {
    position: "relative",
    borderRadius: "10px",
    background: "#12c3f4",
    color: "#fff",

    "&:hover": {
      background: "#65daf8",
    },
  },
  fileUploader: {
    position: "absolute",
    opacity: 0,
    zIndex: 1,
    top: 0,
    left: 0,
    bottom: 0,
    right: 0,
  },
  placeHolder: {
    height: "120px",
  },
}));

type RowType = {
  name: string;
  distance: number;
  duration: number;
  audioSrc: string;
}[];

const Search = () => {
  const classes = useStyles();
  const [rows, setRows] = useState<RowType>([]);
  const [target, setTarget] = useState(null);
  const targetUploadRef = useRef(null!);
  return (
    <div className={classes.searchWrapper}>
      <div className={classes.targetFileWrapper}>
        <h1 className={classes.title}>Target Audio File</h1>
        {target ? <p>target</p> : <div className={classes.placeHolder}></div>}
      </div>

      <div className={classes.searchButton}>
        <Button variant="contained" classes={{ root: classes.button }}>
          Default Target Audio File
          <input
            type="file"
            ref={targetUploadRef}
            className={classes.fileUploader}
          />
        </Button>
      </div>

      <div className={classes.tableWrapper}>
        <SearchResults rows={rows} />
      </div>
    </div>
  );
};
export default Search;
