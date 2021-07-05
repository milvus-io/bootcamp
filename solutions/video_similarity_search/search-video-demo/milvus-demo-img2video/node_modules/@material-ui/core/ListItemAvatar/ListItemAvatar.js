"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = exports.styles = void 0;

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _objectWithoutProperties2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutProperties"));

var _react = _interopRequireDefault(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _clsx = _interopRequireDefault(require("clsx"));

var _withStyles = _interopRequireDefault(require("../styles/withStyles"));

var _ListContext = _interopRequireDefault(require("../List/ListContext"));

var styles = {
  /* Styles applied to the root element. */
  root: {
    minWidth: 56,
    flexShrink: 0
  },

  /* Styles applied to the root element when the parent `ListItem` uses `alignItems="flex-start"`. */
  alignItemsFlexStart: {
    marginTop: 8
  }
};
/**
 * A simple wrapper to apply `List` styles to an `Avatar`.
 */

exports.styles = styles;

var ListItemAvatar = _react.default.forwardRef(function ListItemAvatar(props, ref) {
  var classes = props.classes,
      className = props.className,
      other = (0, _objectWithoutProperties2.default)(props, ["classes", "className"]);

  var context = _react.default.useContext(_ListContext.default);

  return _react.default.createElement("div", (0, _extends2.default)({
    className: (0, _clsx.default)(classes.root, className, context.alignItems === 'flex-start' && classes.alignItemsFlexStart),
    ref: ref
  }, other));
});

process.env.NODE_ENV !== "production" ? ListItemAvatar.propTypes = {
  /**
   * The content of the component – normally `Avatar`.
   */
  children: _propTypes.default.element.isRequired,

  /**
   * Override or extend the styles applied to the component.
   * See [CSS API](#css) below for more details.
   */
  classes: _propTypes.default.object.isRequired,

  /**
   * @ignore
   */
  className: _propTypes.default.string
} : void 0;

var _default = (0, _withStyles.default)(styles, {
  name: 'MuiListItemAvatar'
})(ListItemAvatar);

exports.default = _default;