import { IconButton, makeStyles, Tooltip, Typography } from "@material-ui/core";
import PlayArrowIcon from "@material-ui/icons/PlayArrow";
import PauseIcon from "@material-ui/icons/Pause";
import React, { useRef, useState, useEffect } from "react";
import { formatDuration } from "../utils/common";

const useStyles = makeStyles((theme) => ({
  wrapper: {
    backgroundColor: "#fff",
    height: "80px",
    paddingLeft: theme.spacing(3),
    boxSizing: "border-box",
    display: "flex",
    alignItems: "center",

    "&:hover": {
      backgroundColor: "#f3fcfe",
      outline: "1px solid #12c3f4",

      fontWeight: "bold",
    },
  },
  text: {
    color: "rgba(0, 0, 0, 0.33)",
    marginRight: theme.spacing(3),

    "&:hover": {
      cursor: "default",
    },
  },
  index: {
    color: "rgba(0, 0, 0, 0.33)",
    fontWeight: "bold",
    width: "20px",
    marginRight: theme.spacing(3),
  },
  name: {
    flexGrow: 1,
    lineHeight: "24px",
    fontSize: "16px",
    margin: theme.spacing(0, 3),
    cursor: "default",
    overflow: "hidden",
    whiteSpace: "nowrap",
    textOverflow: "ellipsis",
  },
  nameActive: {
    flexGrow: 1,
    lineHeight: "24px",
    fontSize: "16px",
    fontWeight: "bold",
    marginLeft: theme.spacing(3),
    cursor: "default",
  },
}));
type PropsType = {
  audioSrc: string;
  index: number;
  distance: number;
  duration: number;
  name: string;
};

const AudioPlayer: React.FC<PropsType> = (props) => {
  const classes = useStyles();
  const { audioSrc, index, distance, duration, name } = props;
  const [canPlay, setCanPlay] = useState(false);
  const [isPaused, setIsPaused] = useState(true);
  const audioRef = useRef<HTMLAudioElement>(null!);

  const handleAudioPlayOrPause = () => {
    const audio = audioRef.current;
    setIsPaused(!audio.paused);

    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.addEventListener("canplay", () => {
        setCanPlay(true);
      });

      audio.addEventListener("ended", () => {
        setIsPaused(true);
      });
    }
  }, []);

  return (
    <>
      <audio ref={audioRef} src={audioSrc}></audio>

      <div className={classes.wrapper}>
        {index && <div className={classes.index}>{index}</div>}
        <IconButton disabled={!canPlay} onClick={handleAudioPlayOrPause}>
          {isPaused ? <PlayArrowIcon /> : <PauseIcon />}
        </IconButton>

        <Tooltip title={name} placement="top-start">
          <Typography variant="inherit" className={classes.name}>
            {name || "name"}
          </Typography>
        </Tooltip>

        <Typography variant="body1" className={classes.text}>
          {formatDuration(
            duration || (audioRef.current && audioRef.current.duration)
          )}
        </Typography>

        {distance && (
          <Tooltip title={distance} placement="top">
            <Typography variant="body1" className={classes.text}>
              {distance.toFixed(4)}
            </Typography>
          </Tooltip>
        )}
      </div>
    </>
  );
};

export default AudioPlayer;
