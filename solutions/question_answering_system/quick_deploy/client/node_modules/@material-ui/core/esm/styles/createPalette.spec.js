import { blue, common } from '@material-ui/core/colors';
import { createMuiTheme } from '@material-ui/core/styles';
{
  var palette = createMuiTheme().palette;
  var color = blue;
  var option = {
    main: blue[400]
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
  var themeCommons = common;
}