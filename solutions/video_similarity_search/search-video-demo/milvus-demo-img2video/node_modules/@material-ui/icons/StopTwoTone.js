"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M8 8h8v8H8z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M6 18h12V6H6v12zM8 8h8v8H8V8z"
})), 'StopTwoTone');

exports.default = _default;