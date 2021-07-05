"use strict";

var _styles = require("@material-ui/core/styles");

{
  var theme;
  theme = (0, _styles.createMuiTheme)({
    spacing: 8
  });
  theme = (0, _styles.createMuiTheme)({
    spacing: function spacing(factor) {
      return "".concat(0.8 * factor, "rem");
    }
  });
}