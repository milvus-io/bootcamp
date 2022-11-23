import React, { useState, useRef, useContext } from "react";
import { Theme, makeStyles } from "@material-ui/core/styles";
import SearchResults from "./searchResults";
import AudioPlayer from "./audioPlayer";
import Button from "@material-ui/core/Button";
import { rootContext } from "../context";

const useStyles = makeStyles((theme: Theme) => ({
  searchWrapper: {
    height: "100%",
    padding: "32px",
    boxSizing: "border-box",
  },
  title: {
    lineHeight: "24px",
    padding: "8px 0",
  },
  searchButton: {
    marginBottom: "20px",
  },
  uploadWrapper: {},
  tableWrapper: {
    height: "calc(100vh - 64px - 140px - 56px - 56px)",
  },
  resultWrapper: {
    height: "100%",
  },
  targetFileWrapper: {
    marginBottom: "20px",
  },
  button: {
    position: "relative",
    borderRadius: "10px",
    background: "#12c3f4",
    color: "#fff",
    textTransform: "capitalize",

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
    height: "80px",
  },
}));

type RowType = {
  name: string;
  distance: number;
  audioSrc: string;
}[];

const Search = () => {
  const classes = useStyles();
  const [rows, setRows] = useState<RowType>([]);
  const [target, setTarget] = useState({
    index: 1,
    audioSrc: "",
    name: "",
    distance: 1,
  });
  const targetUploadRef = useRef<HTMLInputElement>(null!);

  const { openSnackbar, setLoading, search, tableName } =
    useContext(rootContext);

  const getSearchedAudio = (file: File) => {
    const name = file.name;
    const src = URL.createObjectURL(file);
    return {
      name,
      src,
    };
  };

  const handleSelectTargetFile = async () => {
    const file = targetUploadRef.current.files![0];

    if (!file) {
      openSnackbar("Please Upload Target File!");
      return;
    }
    const { name, src } = getSearchedAudio(file);
    setTarget({
      index: 1,
      audioSrc: src,
      name: name,
      distance: 1,
    });
    setLoading(true);
    const fd = new FormData();
    fd.append("audio", file);
    fd.append("Table", tableName);

    try {
      const { data, status } = await search(fd);
      if (status === 200) {
        const resData = data.map((item: [string, [string, number]]) => {
          const [audioSrc, [name, distance]] = item;

          return {
            audioSrc,
            duration: 0,
            distance,
            name,
          };
        });

        setRows(resData);
      }
    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={classes.searchWrapper}>
      <div className={classes.targetFileWrapper}>
        <h1 className={classes.title}>Target Audio File</h1>
        {target.name ? (
          <AudioPlayer {...target} index={1} />
        ) : (
          <div className={classes.placeHolder}></div>
        )}
      </div>

      <div className={classes.searchButton}>
        <Button variant="contained" classes={{ root: classes.button }}>
          Default Target Audio File
          <input
            type="file"
            ref={targetUploadRef}
            className={classes.fileUploader}
            onChange={handleSelectTargetFile}
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
