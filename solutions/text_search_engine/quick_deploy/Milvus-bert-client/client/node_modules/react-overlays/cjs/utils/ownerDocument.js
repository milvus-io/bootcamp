"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

exports.__esModule = true;
exports["default"] = _default;

var _reactDom = _interopRequireDefault(require("react-dom"));

var _ownerDocument = _interopRequireDefault(require("dom-helpers/ownerDocument"));

function _default(componentOrElement) {
  return (0, _ownerDocument["default"])(_reactDom["default"].findDOMNode(componentOrElement));
}

module.exports = exports.default;