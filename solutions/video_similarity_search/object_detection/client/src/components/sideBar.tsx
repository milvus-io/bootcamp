import { makeStyles, Theme } from "@material-ui/core";
import BackupIcon from "@material-ui/icons/Backup";
import SearchIcon from "@material-ui/icons/Search";

const useStyles = makeStyles((theme: Theme) => ({
  sideBarWrapper: {
    width: "60px",
    height: "100%",
    background: "#fff",
    display: "flex",
    flexDirection: "column",

    [theme.breakpoints.down(theme.breakpoints.values.sm)]: {
      width: "100%",
      height: "60px",
      flexDirection: "row",
    },
  },
  iconWrapper: {
    width: "60px",
    height: "60px",
    fontSize: "36px",
    padding: "12px",
    cursor: "pointer",
    border: "none",
    outline: "none",
    background: "#fff",

    "&.active": {
      background: "#F3FCFE",
      color: "#12C3F4",
    },
  },
}));

type Props = {
  activeItem: "upload" | "search";
  setActiveItem: React.Dispatch<React.SetStateAction<"upload" | "search">>;
  isDataReady: boolean;
};

const SideBar = (props: Props): JSX.Element => {
  const classes = useStyles();
  const { activeItem, setActiveItem, isDataReady } = props;

  const handleClick = (val: "upload" | "search") => {
    if (activeItem === val) return;
    setActiveItem(val);
  };
  return (
    <div className={classes.sideBarWrapper}>
      <button
        className={`${classes.iconWrapper} ${
          activeItem === "upload" ? "active" : ""
        }`}
        onClick={() => handleClick("upload")}
      >
        <BackupIcon />
      </button>
      <button
        className={`${classes.iconWrapper} ${
          activeItem === "search" ? "active" : ""
        }`}
        onClick={() => handleClick("search")}
        disabled={!isDataReady}
      >
        <SearchIcon />
      </button>
    </div>
  );
};

export default SideBar;
