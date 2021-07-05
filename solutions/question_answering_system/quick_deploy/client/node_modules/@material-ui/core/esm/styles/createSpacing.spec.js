import { createMuiTheme } from '@material-ui/core/styles';
{
  var theme;
  theme = createMuiTheme({
    spacing: 8
  });
  theme = createMuiTheme({
    spacing: function spacing(factor) {
      return "".concat(0.8 * factor, "rem");
    }
  });
}