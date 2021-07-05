"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _objectWithoutProperties2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutProperties"));

var _react = _interopRequireDefault(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _withStyles = _interopRequireDefault(require("../styles/withStyles"));

var _colorManipulator = require("../styles/colorManipulator");

var _MoreHoriz = _interopRequireDefault(require("../internal/svg-icons/MoreHoriz"));

var styles = function styles(theme) {
  return {
    root: {
      display: 'flex'
    },
    icon: {
      width: 24,
      height: 16,
      backgroundColor: theme.palette.grey[100],
      color: theme.palette.grey[700],
      borderRadius: 2,
      marginLeft: theme.spacing(0.5),
      marginRight: theme.spacing(0.5),
      cursor: 'pointer',
      '&:hover, &:focus': {
        backgroundColor: theme.palette.grey[200]
      },
      '&:active': {
        boxShadow: theme.shadows[0],
        backgroundColor: (0, _colorManipulator.emphasize)(theme.palette.grey[200], 0.12)
      }
    }
  };
};
/**
 * @ignore - internal component.
 */


function BreadcrumbCollapsed(props) {
  var classes = props.classes,
      other = (0, _objectWithoutProperties2.default)(props, ["classes"]);
  return _react.default.createElement("li", (0, _extends2.default)({
    className: classes.root
  }, other), _react.default.createElement(_MoreHoriz.default, {
    className: classes.icon
  }));
}

process.env.NODE_ENV !== "production" ? BreadcrumbCollapsed.propTypes = {
  /**
   * @ignore
   */
  classes: _propTypes.default.object.isRequired
} : void 0;

var _default = (0, _withStyles.default)(styles, {
  name: 'PrivateBreadcrumbCollapsed'
})(BreadcrumbCollapsed);

exports.default = _default;