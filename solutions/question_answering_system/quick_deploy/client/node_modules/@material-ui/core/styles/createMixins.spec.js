"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _styles = require("@material-ui/core/styles");

{
  var theme = (0, _styles.createMuiTheme)({
    mixins: {
      toolbar: {
        background: '#fff',
        minHeight: 36,
        '@media (min-width:0px) and (orientation: landscape)': {
          minHeight: 24
        },
        '@media (min-width:600px)': {
          minHeight: 48
        }
      }
    }
  });
  var useStyles = (0, _styles.makeStyles)(function (theme) {
    return {
      appBarSpacer: theme.mixins.toolbar,
      toolbarIcon: (0, _extends2.default)({
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        padding: '0 8px'
      }, theme.mixins.toolbar)
    };
  });
}