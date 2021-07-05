import { createMuiTheme } from '@material-ui/core/styles';
{
  let theme;
  theme = createMuiTheme({
    spacing: 8
  });
  theme = createMuiTheme({
    spacing: factor => `${0.8 * factor}rem`
  });
}