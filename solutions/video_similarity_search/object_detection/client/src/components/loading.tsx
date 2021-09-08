import HourglassEmptyIcon from "@material-ui/icons/HourglassEmpty";
import { Typography, makeStyles, Theme } from "@material-ui/core";

type PropsType = {
  open: boolean;
  className?: string;
};
const useStyles = makeStyles((theme: Theme) => ({
  loadingWrapper: {
    position: "fixed",
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    background: "rgba(0,0,0,0.5)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    color: "#fff",
  },
  content: {
    textAlign: "center",
  },
  msg: {
    marginTop: theme.spacing(1),
  },
}));
const GlobalLoading = (props: PropsType): JSX.Element => {
  const { open, className = "" } = props;
  const classes = useStyles();

  return (
    <>
      {open ? (
        <div className={`${classes.loadingWrapper} ${className}`}>
          <div className={classes.content}>
            <HourglassEmptyIcon />
            <Typography component="p" className={classes.msg}>
              Loading...
            </Typography>
          </div>
        </div>
      ) : null}
    </>
  );
};

export default GlobalLoading;
