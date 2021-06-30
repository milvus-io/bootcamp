import { createMuiTheme } from '@material-ui/core/styles';
{
  // properties of the variants can be "unset"
  var theme = createMuiTheme({
    typography: {
      allVariants: {
        fontStyle: undefined
      }
    }
  }); // $ExpectType string | undefined

  var maybeFontStyle = theme.typography.body1.fontStyle;
}