"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M15 9.86v4.28L18.03 12zM6 9.86v4.28L9.03 12z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M4 18l8.5-6L4 6v12zm2-8.14L9.03 12 6 14.14V9.86zM21.5 12L13 6v12l8.5-6zM15 9.86L18.03 12 15 14.14V9.86z"
})), 'FastForwardTwoTone');

exports.default = _default;