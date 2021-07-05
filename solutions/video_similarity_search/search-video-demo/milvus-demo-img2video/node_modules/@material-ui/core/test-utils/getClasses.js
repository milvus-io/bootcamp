"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = getClasses;

var _react = _interopRequireDefault(require("react"));

var _createShallow = _interopRequireDefault(require("./createShallow"));

var shallow = (0, _createShallow.default)(); // Helper function to extract the classes from a styleSheet.

function getClasses(element) {
  var useStyles = element.type.useStyles;
  var classes;

  function Listener() {
    classes = useStyles(element.props);
    return null;
  }

  shallow(_react.default.createElement(Listener, null));
  return classes;
}