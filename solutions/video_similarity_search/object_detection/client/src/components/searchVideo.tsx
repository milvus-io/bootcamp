import { Button, Grid, makeStyles, Typography, Theme } from "@material-ui/core";
import VideoPlayer from "./videoPlayer";
import { useState, useRef } from "react";
import { DEMOS_BASE_URL } from "../http/constants";

const useStyles = makeStyles((theme: Theme) => ({
  audioWrapper: {
    display: "flex",
    justifyContent: "center",
    alignItems: "flex-start",

    height: "100%",
    width: "100%",
    color: "#000",

    [theme.breakpoints.down(theme.breakpoints.values.sm)]: {
      alignItems: "center",
      justifyContent: "center",
    },
  },
  search: {},
  title: {
    fontSize: "16px",
    fontWeight: "bold",
    marginBottom: theme.spacing(2),
    textTransform: "uppercase",
    color: "rgba(0, 0, 0, 0.6)",
  },
  btnWrapper: {
    marginTop: "42px",
    "& .download-btn": {
      color: "rgba(0, 0, 0, 0.33)",
      textDecoration: "none",
    },
  },
  result: {},
  resultHeader: {
    display: "flex",
    marginTop: "5px",
    // marginBottom: theme.spacing(2),
    color: "rgba(0, 0, 0, 0.33)",
    paddingBottom: theme.spacing(2),
    borderBottom: "1px solid rgba(0, 0, 0, 0.15)",
  },
  index: {
    fontSize: "12.8px",
    lineHeight: "15px",
    marginRight: theme.spacing(3),
    width: "20px",
  },
  image: {
    flex: "0 0 80px",
    marginRight: theme.spacing(2),
    "& img": {
      width: "80px",
    },
  },
  name: {
    fontSize: "12.8px",
    fontWeight: "bold",
    flex: "0 0 200px",
    marginRight: theme.spacing(2),
  },
  time: {
    fontSize: "12.8px",
    flex: 1,
    textAlign: "center",
    maxWidth: "60em !important",
  },
  resultWrapper: {
    overflowY: "auto",
    maxHeight: `calc(100vh - 64px - 48px - 32px - 24px - 24px)`,
  },
  resultItem: {
    display: "flex",
    marginTop: theme.spacing(1),
  },
  noData: {
    marginTop: theme.spacing(4),
    display: "flex",
    justifyContent: "center",
  },
}));

const SearchVideo = () => {
  const classes = useStyles();
  const [player, setPlayer] = useState(null);
  const videoUploadRef = useRef(null);
  const [searchedVideo, setSearchedVideo] = useState("");
  const [showResult, setShowResult] = useState([]);

  const handleUploadVideo = () => {};

  return (
    <Grid container spacing={2} className={classes.audioWrapper}>
      <Grid item xs={10} md={6} lg={5} className={classes.search}>
        <Typography variant="body1" className={classes.title}>
          What is it
        </Typography>

        <VideoPlayer
          setPlayer={setPlayer}
          player={player}
          source={searchedVideo}
        />

        <div className={classes.btnWrapper}>
          <Button variant="contained" onClick={handleUploadVideo}>
            Upload a video to search
          </Button>
          <Button>
            <a
              href={`${DEMOS_BASE_URL}/video2/download_obj_video`}
              className="download-btn"
            >
              download an example video
            </a>
          </Button>
        </div>
      </Grid>
      <Grid item xs={10} md={6} lg={7} className={classes.result}>
        <div className={classes.resultHeader}>
          <Typography variant="inherit" className={classes.image}>
            Image
          </Typography>
          <Typography variant="body1" className={classes.name}>
            Name
          </Typography>
        </div>
        <div className={classes.resultWrapper}>
          {showResult.length > 0 ? (
            showResult.map((v, index) => (
              <div className={classes.resultItem} key={index}>
                <div className={classes.image}>
                  <img alt="face" src={v[1]}></img>
                </div>
                <Typography variant="body1" className={classes.name}>
                  {v[0]}
                </Typography>
              </div>
            ))
          ) : (
            <div className={classes.noData}>No data available</div>
          )}
        </div>
      </Grid>
    </Grid>
  );
};

export default SearchVideo;
