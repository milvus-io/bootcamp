import { blue, common } from '@material-ui/core/colors';
import { createMuiTheme } from '@material-ui/core/styles';
{
  const palette = createMuiTheme().palette;
  const color = blue;
  const option = {
    main: blue[400]
  };
  const colorOrOption = undefined;
  palette.augmentColor(color);
  palette.augmentColor(color, 400);
  palette.augmentColor(color, 400, 200, 600);
  palette.augmentColor(color, 400, undefined, 600);
  palette.augmentColor(option);
  palette.augmentColor(option, 400); // $ExpectError

  palette.augmentColor(colorOrOption);
  palette.augmentColor(colorOrOption, 400); // $ExpectError

  const augmentedColor = palette.augmentColor(colorOrOption);
}
{
  const themeCommons = common;
}