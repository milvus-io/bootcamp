"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  fillOpacity: ".36",
  d: "M2 20h20c1.1 0 2 .9 2 2s-.9 2-2 2H2c-1.1 0-2-.9-2-2s.9-2 2-2z"
}), _react.default.createElement("path", {
  d: "M10.63 3.93L6.06 15.58c-.27.68.23 1.42.97 1.42.43 0 .82-.27.98-.68L8.87 14h6.25l.87 2.32c.15.41.54.68.98.68.73 0 1.24-.74.97-1.42L13.37 3.93C13.14 3.37 12.6 3 12 3c-.6 0-1.15.37-1.37.93zM9.62 12L12 5.67 14.38 12H9.62z"
})), 'FormatColorTextRounded');

exports.default = _default;