"use strict";

var _colors = require("@material-ui/core/colors");

var _styles = require("@material-ui/core/styles");

{
  var palette = (0, _styles.createMuiTheme)().palette;
  var color = _colors.blue;
  var option = {
    main: _colors.blue[400]
  };
  var colorOrOption = undefined;
  palette.augmentColor(color);
  palette.augmentColor(color, 400);
  palette.augmentColor(color, 400, 200, 600);
  palette.augmentColor(color, 400, undefined, 600);
  palette.augmentColor(option);
  palette.augmentColor(option, 400); // $ExpectError

  palette.augmentColor(colorOrOption);
  palette.augmentColor(colorOrOption, 400); // $ExpectError

  var augmentedColor = palette.augmentColor(colorOrOption);
}
{
  var themeCommons = _colors.common;
}