"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = exports.styles = void 0;

var _objectWithoutProperties2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutProperties"));

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _react = _interopRequireDefault(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _clsx = _interopRequireDefault(require("clsx"));

var _styles = require("@material-ui/core/styles");

var _ButtonBase = _interopRequireDefault(require("@material-ui/core/ButtonBase"));

var _utils = require("@material-ui/core/utils");

// @inheritedComponent ButtonBase
var styles = function styles(theme) {
  return {
    /* Styles applied to the root element. */
    root: (0, _extends2.default)({}, theme.typography.button, {
      boxSizing: 'border-box',
      borderRadius: theme.shape.borderRadius,
      height: 48,
      padding: '0px 12px',
      border: "1px solid ".concat((0, _styles.fade)(theme.palette.action.active, 0.12)),
      color: (0, _styles.fade)(theme.palette.action.active, 0.38),
      '&$selected': {
        color: theme.palette.action.active,
        backgroundColor: (0, _styles.fade)(theme.palette.action.active, 0.12),
        '&:hover': {
          backgroundColor: (0, _styles.fade)(theme.palette.action.active, 0.15)
        },
        '& + &': {
          borderLeft: 0,
          marginLeft: 0
        }
      },
      '&$disabled': {
        color: (0, _styles.fade)(theme.palette.action.disabled, 0.12)
      },
      '&:hover': {
        textDecoration: 'none',
        // Reset on mouse devices
        backgroundColor: (0, _styles.fade)(theme.palette.text.primary, 0.05),
        '@media (hover: none)': {
          backgroundColor: 'transparent'
        },
        '&$disabled': {
          backgroundColor: 'transparent'
        }
      }
    }),

    /* Pseudo-class applied to the root element if `disabled={true}`. */
    disabled: {},

    /* Pseudo-class applied to the root element if `selected={true}`. */
    selected: {},

    /* Styles applied to the `label` wrapper element. */
    label: {
      width: '100%',
      // Ensure the correct width for iOS Safari
      display: 'inherit',
      alignItems: 'inherit',
      justifyContent: 'inherit'
    },

    /* Styles applied to the root element if `size="small"`. */
    sizeSmall: {
      height: 40,
      padding: '0px 8px',
      fontSize: theme.typography.pxToRem(13)
    },

    /* Styles applied to the root element if `size="large"`. */
    sizeLarge: {
      height: 56,
      padding: '0px 16px',
      fontSize: theme.typography.pxToRem(15)
    }
  };
};

exports.styles = styles;

var ToggleButton = _react.default.forwardRef(function ToggleButton(props, ref) {
  var children = props.children,
      classes = props.classes,
      className = props.className,
      _props$disabled = props.disabled,
      disabled = _props$disabled === void 0 ? false : _props$disabled,
      _props$disableFocusRi = props.disableFocusRipple,
      disableFocusRipple = _props$disableFocusRi === void 0 ? false : _props$disableFocusRi,
      _props$disableRipple = props.disableRipple,
      disableRipple = _props$disableRipple === void 0 ? false : _props$disableRipple,
      onChange = props.onChange,
      onClick = props.onClick,
      selected = props.selected,
      _props$size = props.size,
      size = _props$size === void 0 ? 'medium' : _props$size,
      value = props.value,
      other = (0, _objectWithoutProperties2.default)(props, ["children", "classes", "className", "disabled", "disableFocusRipple", "disableRipple", "onChange", "onClick", "selected", "size", "value"]);

  var handleChange = function handleChange(event) {
    if (onClick) {
      onClick(event, value);

      if (event.isDefaultPrevented()) {
        return;
      }
    }

    if (onChange) {
      onChange(event, value);
    }
  };

  return _react.default.createElement(_ButtonBase.default, (0, _extends2.default)({
    className: (0, _clsx.default)(classes.root, className, disabled && classes.disabled, selected && classes.selected, size !== 'medium' && classes["size".concat((0, _utils.capitalize)(size))]),
    disabled: disabled,
    focusRipple: !disableFocusRipple,
    ref: ref,
    onClick: handleChange,
    onChange: onChange,
    value: value,
    disableRipple: disableRipple,
    "aria-pressed": selected
  }, other), _react.default.createElement("span", {
    className: classes.label
  }, children));
});

process.env.NODE_ENV !== "production" ? ToggleButton.propTypes = {
  /**
   * The content of the button.
   */
  children: _propTypes.default.node.isRequired,

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
   * If `true`, the button will be disabled.
   */
  disabled: _propTypes.default.bool,

  /**
   * If `true`, the  keyboard focus ripple will be disabled.
   * `disableRipple` must also be true.
   */
  disableFocusRipple: _propTypes.default.bool,

  /**
   * If `true`, the ripple effect will be disabled.
   */
  disableRipple: _propTypes.default.bool,

  /**
   * @ignore
   */
  onChange: _propTypes.default.func,

  /**
   * @ignore
   */
  onClick: _propTypes.default.func,

  /**
   * If `true`, the button will be rendered in an active state.
   */
  selected: _propTypes.default.bool,

  /**
   * @ignore
   */
  size: _propTypes.default.oneOf(['small', 'medium', 'large']),

  /**
   * The value to associate with the button when selected in a
   * ToggleButtonGroup.
   */
  value: _propTypes.default.any.isRequired
} : void 0;

var _default = (0, _styles.withStyles)(styles, {
  name: 'MuiToggleButton'
})(ToggleButton);

exports.default = _default;