import { makeStyles, Theme } from "@material-ui/core";

export const useHomeStyles = makeStyles((theme: Theme) => ({
  root: {
    flex: 1,
    display: "flex",
    background: "#fafafa",

    [theme.breakpoints.down(theme.breakpoints.values.sm)]: {
      flexDirection: "column",
    },
  },
  content: {
    flex: 1,
  },
  input: {
    visibility: "hidden",
    opacity: 0,
  },
}));
