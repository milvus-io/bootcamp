"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M16 12.5h3V16h-3zM6 7h3v9H6zm5 5.5h3V16h-3zM11 7h8v3.5h-8z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M4 5v13h17V5H4zm5 11H6V7h3v9zm5 0h-3v-3.5h3V16zm5 0h-3v-3.5h3V16zm0-5.5h-8V7h8v3.5z"
})), 'ViewQuiltTwoTone');

exports.default = _default;