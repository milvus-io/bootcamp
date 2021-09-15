import {
  Player,
  ControlBar,
  ReplayControl,
  ForwardControl,
  CurrentTimeDisplay,
  TimeDivider,
  PlaybackRateMenuButton,
  VolumeMenuButton,
  BigPlayButton,
} from "video-react";
import "video-react/dist/video-react.css";
import { IconButton, makeStyles } from "@material-ui/core";
import FaceIcon from "@material-ui/icons/Face";
import { useEffect } from "react";

const useStyles = makeStyles((theme) => ({
  root: {
    position: "relative",
  },
  face: {
    position: "absolute",
    top: "10px",
    right: "10px",
    backgroundColor: "#12C3F4 !important",
    color: "#fff",
  },
  placeholder: {
    width: "100%",
    height: "210px",
    padding: "0 80px",
    boxSizing: "border-box",
    backgroundColor: "#000",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#fff",
    fontWeight: "bold",
    fontSize: "20px",
    textAlign: "center",
    lineHeight: "32px",
  },
}));

const VideoPlayer = (props) => {
  const { setPlayer, player, showBtn, source, cb } = props;
  const classes = useStyles();
  const handleClick = async () => {
    const { player: myPlayer } = player.getState();
    cb(myPlayer.currentTime);
  };
  useEffect(() => {
    if (!source || !player) return;
    player.load();
  }, [source, player]);


  return (
    <div className={classes.root}>
      {source ? (
        <Player
          ref={(player) => {
            setPlayer(player);
          }}
        >
          <source src={source} />
          <ControlBar>
            <ReplayControl seconds={5} order={1.1} />
            <ForwardControl seconds={5} order={1.2} />
            <CurrentTimeDisplay order={4.1} />
            <TimeDivider order={4.2} />
            <PlaybackRateMenuButton rates={[5, 2, 1, 0.5, 0.1]} order={7.1} />
            <VolumeMenuButton />
          </ControlBar>
          <BigPlayButton position="center" />
        </Player>
      ) : (
        <div className={classes.placeholder}>
          You can search an avi file with a maxium file size of 20MB
        </div>
      )}

      {showBtn && (
        <IconButton className={classes.face} onClick={handleClick}>
          <FaceIcon />
        </IconButton>
      )}
    </div>
  );
};

export default VideoPlayer;
