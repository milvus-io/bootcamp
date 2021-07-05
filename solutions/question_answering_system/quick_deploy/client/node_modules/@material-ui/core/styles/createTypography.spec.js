"use strict";

var _styles = require("@material-ui/core/styles");

{
  // properties of the variants can be "unset"
  var theme = (0, _styles.createMuiTheme)({
    typography: {
      allVariants: {
        fontStyle: undefined
      }
    }
  }); // $ExpectType string | undefined

  var maybeFontStyle = theme.typography.body1.fontStyle;
}