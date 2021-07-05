"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = exports.styles = void 0;

var _react = _interopRequireDefault(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _clsx = _interopRequireDefault(require("clsx"));

var _CheckCircle = _interopRequireDefault(require("../internal/svg-icons/CheckCircle"));

var _Warning = _interopRequireDefault(require("../internal/svg-icons/Warning"));

var _withStyles = _interopRequireDefault(require("../styles/withStyles"));

var _SvgIcon = _interopRequireDefault(require("../SvgIcon"));

var styles = function styles(theme) {
  return {
    /* Styles applied to the root element. */
    root: {
      display: 'block',
      color: theme.palette.text.disabled,
      '&$completed': {
        color: theme.palette.primary.main
      },
      '&$active': {
        color: theme.palette.primary.main
      },
      '&$error': {
        color: theme.palette.error.main
      }
    },

    /* Styles applied to the SVG text element. */
    text: {
      fill: theme.palette.primary.contrastText,
      fontSize: theme.typography.caption.fontSize,
      fontFamily: theme.typography.fontFamily
    },

    /* Pseudo-class applied to the root element if `active={true}`. */
    active: {},

    /* Pseudo-class applied to the root element if `completed={true}`. */
    completed: {},

    /* Pseudo-class applied to the root element if `error={true}`. */
    error: {}
  };
};

exports.styles = styles;

var _ref = _react.default.createElement("circle", {
  cx: "12",
  cy: "12",
  r: "12"
});

var StepIcon = _react.default.forwardRef(function StepIcon(props, ref) {
  var _props$completed = props.completed,
      completed = _props$completed === void 0 ? false : _props$completed,
      icon = props.icon,
      _props$active = props.active,
      active = _props$active === void 0 ? false : _props$active,
      _props$error = props.error,
      error = _props$error === void 0 ? false : _props$error,
      classes = props.classes;

  if (typeof icon === 'number' || typeof icon === 'string') {
    var className = (0, _clsx.default)(classes.root, active && classes.active, error && classes.error, completed && classes.completed);

    if (error) {
      return _react.default.createElement(_Warning.default, {
        className: className,
        ref: ref
      });
    }

    if (completed) {
      return _react.default.createElement(_CheckCircle.default, {
        className: className,
        ref: ref
      });
    }

    return _react.default.createElement(_SvgIcon.default, {
      className: className,
      ref: ref
    }, _ref, _react.default.createElement("text", {
      className: classes.text,
      x: "12",
      y: "16",
      textAnchor: "middle"
    }, icon));
  }

  return icon;
});

process.env.NODE_ENV !== "production" ? StepIcon.propTypes = {
  /**
   * Whether this step is active.
   */
  active: _propTypes.default.bool,

  /**
   * Override or extend the styles applied to the component.
   * See [CSS API](#css) below for more details.
   */
  classes: _propTypes.default.object.isRequired,

  /**
   * Mark the step as completed. Is passed to child components.
   */
  completed: _propTypes.default.bool,

  /**
   * Mark the step as failed.
   */
  error: _propTypes.default.bool,

  /**
   * The label displayed in the step icon.
   */
  icon: _propTypes.default.node.isRequired
} : void 0;

var _default = (0, _withStyles.default)(styles, {
  name: 'MuiStepIcon'
})(StepIcon);

exports.default = _default;