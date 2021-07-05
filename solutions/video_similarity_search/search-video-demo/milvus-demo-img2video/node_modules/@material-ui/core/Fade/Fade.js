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

var _reactTransitionGroup = require("react-transition-group");

var _transitions = require("../styles/transitions");

var _useTheme = _interopRequireDefault(require("../styles/useTheme"));

var _utils = require("../transitions/utils");

var _useForkRef = _interopRequireDefault(require("../utils/useForkRef"));

var styles = {
  entering: {
    opacity: 1
  },
  entered: {
    opacity: 1
  }
};
var defaultTimeout = {
  enter: _transitions.duration.enteringScreen,
  exit: _transitions.duration.leavingScreen
};
/**
 * The Fade transition is used by the [Modal](/components/modal/) component.
 * It uses [react-transition-group](https://github.com/reactjs/react-transition-group) internally.
 */

var Fade = _react.default.forwardRef(function Fade(props, ref) {
  var children = props.children,
      inProp = props.in,
      onEnter = props.onEnter,
      onExit = props.onExit,
      style = props.style,
      _props$timeout = props.timeout,
      timeout = _props$timeout === void 0 ? defaultTimeout : _props$timeout,
      other = (0, _objectWithoutProperties2.default)(props, ["children", "in", "onEnter", "onExit", "style", "timeout"]);
  var theme = (0, _useTheme.default)();
  var handleRef = (0, _useForkRef.default)(children.ref, ref);

  var handleEnter = function handleEnter(node, isAppearing) {
    (0, _utils.reflow)(node); // So the animation always start from the start.

    var transitionProps = (0, _utils.getTransitionProps)({
      style: style,
      timeout: timeout
    }, {
      mode: 'enter'
    });
    node.style.webkitTransition = theme.transitions.create('opacity', transitionProps);
    node.style.transition = theme.transitions.create('opacity', transitionProps);

    if (onEnter) {
      onEnter(node, isAppearing);
    }
  };

  var handleExit = function handleExit(node) {
    var transitionProps = (0, _utils.getTransitionProps)({
      style: style,
      timeout: timeout
    }, {
      mode: 'exit'
    });
    node.style.webkitTransition = theme.transitions.create('opacity', transitionProps);
    node.style.transition = theme.transitions.create('opacity', transitionProps);

    if (onExit) {
      onExit(node);
    }
  };

  return _react.default.createElement(_reactTransitionGroup.Transition, (0, _extends2.default)({
    appear: true,
    in: inProp,
    onEnter: handleEnter,
    onExit: handleExit,
    timeout: timeout
  }, other), function (state, childProps) {
    return _react.default.cloneElement(children, (0, _extends2.default)({
      style: (0, _extends2.default)({
        opacity: 0,
        visibility: state === 'exited' && !inProp ? 'hidden' : undefined
      }, styles[state], {}, style, {}, children.props.style),
      ref: handleRef
    }, childProps));
  });
});

process.env.NODE_ENV !== "production" ? Fade.propTypes = {
  /**
   * A single child content element.
   */
  children: _propTypes.default.element,

  /**
   * If `true`, the component will transition in.
   */
  in: _propTypes.default.bool,

  /**
   * @ignore
   */
  onEnter: _propTypes.default.func,

  /**
   * @ignore
   */
  onExit: _propTypes.default.func,

  /**
   * @ignore
   */
  style: _propTypes.default.object,

  /**
   * The duration for the transition, in milliseconds.
   * You may specify a single timeout for all transitions, or individually with an object.
   */
  timeout: _propTypes.default.oneOfType([_propTypes.default.number, _propTypes.default.shape({
    enter: _propTypes.default.number,
    exit: _propTypes.default.number
  })])
} : void 0;
var _default = Fade;
exports.default = _default;