import { Button, Grid, makeStyles, Typography, Theme } from "@material-ui/core";
import VideoPlayer from "./videoPlayer";
import { useState, useRef, useContext, useMemo, useEffect } from "react";
import { videoSearch } from "../http";
import { rootContext } from "../context/rootProvider";
import { IVideoSearchResult } from "../types";

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
  hiddenBtn: {
    visibility: "hidden",
    opacity: 0,
    position: "absolute",
    zIndex: -1,
  },
}));

const SearchVideo = () => {
  const classes = useStyles();
  const { setGlobalLoading, openSnackbar } = useContext(rootContext);
  const [player, setPlayer] = useState<any>(null!);
  const videoUploadRef = useRef<HTMLInputElement>(null!);
  const [searchedVideo, setSearchedVideo] = useState("");
  const [searchResult, setSearchResult] = useState<IVideoSearchResult[]>([]);
  const [currentTime, setCurrentTime] = useState(0);

  const handleUploadVideo = () => {
    videoUploadRef.current.click();
  };

  const upload = async () => {
    const fd = new FormData();
    const file = videoUploadRef.current.files![0];
    fd.append("video", file);
    setGlobalLoading(true);

    try {
      const { data } = await videoSearch(fd);
      setSearchResult(data.slice(1));
      setSearchedVideo(data[0]);
    } catch (error) {
      console.log(error);
      openSnackbar("Search failed", "error");
    } finally {
      setGlobalLoading(false);
    }
  };

  const showResult = useMemo(() => {
    return searchResult.filter(
      (v) => parseInt(v.time) === Math.ceil(currentTime)
    );
  }, [currentTime, searchResult]);

  useEffect(() => {
    if (!player) return;
    player.subscribeToStateChange((state: any, prevState: any) => {
      const { currentTime } = state;
      if (currentTime > 0) {
        setCurrentTime(Math.ceil(currentTime));
      }
    });
  }, [player]);

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
          <input
            type="file"
            accept="avi"
            ref={videoUploadRef}
            className={classes.hiddenBtn}
            onChange={upload}
          />
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
          <Typography variant="body1" className={classes.name}>
            Distance
          </Typography>
        </div>
        <div className={classes.resultWrapper}>
          {showResult.length > 0 ? (
            showResult.map((v, index) => (
              <div className={classes.resultItem} key={index}>
                <div className={classes.image}>
                  <img alt="face" src={v.image}></img>
                </div>
                <Typography variant="body1" className={classes.name}>
                  {v.object}
                </Typography>
                <Typography variant="body1" className={classes.name}>
                  {v.distance.toFixed(7)}
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
