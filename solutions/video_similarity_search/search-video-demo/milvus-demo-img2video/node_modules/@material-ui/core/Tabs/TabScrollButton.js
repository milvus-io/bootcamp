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

var _KeyboardArrowLeft = _interopRequireDefault(require("../internal/svg-icons/KeyboardArrowLeft"));

var _KeyboardArrowRight = _interopRequireDefault(require("../internal/svg-icons/KeyboardArrowRight"));

var _withStyles = _interopRequireDefault(require("../styles/withStyles"));

var _ButtonBase = _interopRequireDefault(require("../ButtonBase"));

/* eslint-disable jsx-a11y/aria-role */
var styles = {
  root: {
    width: 40,
    flexShrink: 0
  },
  vertical: {
    width: '100%',
    height: 40,
    '& svg': {
      transform: 'rotate(90deg)'
    }
  }
};
/**
 * @ignore - internal component.
 */

exports.styles = styles;

var _ref = _react.default.createElement(_KeyboardArrowLeft.default, {
  fontSize: "small"
});

var _ref2 = _react.default.createElement(_KeyboardArrowRight.default, {
  fontSize: "small"
});

var TabScrollButton = _react.default.forwardRef(function TabScrollButton(props, ref) {
  var classes = props.classes,
      classNameProp = props.className,
      direction = props.direction,
      orientation = props.orientation,
      visible = props.visible,
      other = (0, _objectWithoutProperties2.default)(props, ["classes", "className", "direction", "orientation", "visible"]);
  var className = (0, _clsx.default)(classes.root, classNameProp, orientation === 'vertical' && classes.vertical);

  if (!visible) {
    return _react.default.createElement("div", {
      className: className
    });
  }

  return _react.default.createElement(_ButtonBase.default, (0, _extends2.default)({
    component: "div",
    className: className,
    ref: ref,
    role: null,
    tabIndex: null
  }, other), direction === 'left' ? _ref : _ref2);
});

process.env.NODE_ENV !== "production" ? TabScrollButton.propTypes = {
  /**
   * Override or extend the styles applied to the component.
   * See [CSS API](#css) below for more details.
   */
  classes: _propTypes.default.object.isRequired,

  /**
   * @ignore
   */
  className: _propTypes.default.string,

  /**
   * Which direction should the button indicate?
   */
  direction: _propTypes.default.oneOf(['left', 'right']).isRequired,

  /**
   * The tabs orientation (layout flow direction).
   */
  orientation: _propTypes.default.oneOf(['horizontal', 'vertical']).isRequired,

  /**
   * Should the button be present or just consume space.
   */
  visible: _propTypes.default.bool.isRequired
} : void 0;

var _default = (0, _withStyles.default)(styles, {
  name: 'PrivateTabScrollButton'
})(TabScrollButton);

exports.default = _default;