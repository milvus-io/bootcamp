import { createStyles, Theme, makeStyles } from "@material-ui/core/styles";

export const createStyle = (style: any) => {
  const useStyles = makeStyles((theme: Theme) => createStyles(style));
  return useStyles({});
};
