"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

exports.__esModule = true;
exports.useDropdownToggle = useDropdownToggle;
exports["default"] = void 0;

var _propTypes = _interopRequireDefault(require("prop-types"));

var _react = require("react");

var _DropdownContext = _interopRequireDefault(require("./DropdownContext"));

/**
 * Wires up Dropdown toggle functinality, returning a set a props to attach
 * to the element that functions as the dropdown toggle (generally a button).
 */
function useDropdownToggle() {
  var _useContext = (0, _react.useContext)(_DropdownContext["default"]),
      show = _useContext.show,
      toggle = _useContext.toggle,
      setToggle = _useContext.setToggle;

  return [{
    ref: setToggle,
    'aria-haspopup': true,
    'aria-expanded': !!show
  }, {
    show: show,
    toggle: toggle
  }];
}

var propTypes = {
  /**
   * A render prop that returns a Toggle element. The `props`
   * argument should spread through to **a component that can accept a ref**. Use
   * the `onToggle` argument to toggle the menu open or closed
   *
   * @type {Function ({
   *   show: boolean,
   *   toggle: (show: boolean) => void,
   *   props: {
   *     ref: (?HTMLElement) => void,
   *     aria-haspopup: true
   *     aria-expanded: boolean
   *   },
   * }) => React.Element}
   */
  children: _propTypes["default"].func.isRequired
};

function DropdownToggle(_ref) {
  var children = _ref.children;

  var _useDropdownToggle = useDropdownToggle(),
      props = _useDropdownToggle[0],
      _useDropdownToggle$ = _useDropdownToggle[1],
      show = _useDropdownToggle$.show,
      toggle = _useDropdownToggle$.toggle;

  return children({
    show: show,
    toggle: toggle,
    props: props
  });
}

DropdownToggle.displayName = 'ReactOverlaysDropdownToggle';
DropdownToggle.propTypes = propTypes;
/** @component */

var _default = DropdownToggle;
exports["default"] = _default;