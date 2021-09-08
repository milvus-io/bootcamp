import { makeStyles, Theme, Typography } from "@material-ui/core";

const useStyles = makeStyles((theme: Theme) => ({
  headerWrapper: {
    width: "100%",
    height: "60px",
    padding: theme.spacing(2, 5),
    background: "#000",
    color: "#fff",
    display: "flex",
    justifyContent: "flex-start",
    alignItems: "flex-end",
    boxSizing: "border-box",

    [theme.breakpoints.down(theme.breakpoints.values.sm)]: {
      padding: theme.spacing(2, 2),
    },
  },
  title: {
    fontSize: "24px",
    lineHeight: "28px",
    fontWeight: "bold",
    marginRight: theme.spacing(3),

    [theme.breakpoints.down(theme.breakpoints.values.sm)]: {
      fontSize: "20px",
      lineHeight: "24px",
      marginRight: theme.spacing(2),
    },
  },
  text: {
    fontSize: "12px",
    lineHeight: "15px",
    textTransform: "uppercase",
  },
}));

const Header = () => {
  const classes = useStyles();
  return (
    <div className={classes.headerWrapper}>
      <Typography component="h1" className={classes.title}>
        Object Detection
      </Typography>
      <Typography component="p" className={classes.text}>
        Powered by milvus
      </Typography>
    </div>
  );
};

export default Header;
