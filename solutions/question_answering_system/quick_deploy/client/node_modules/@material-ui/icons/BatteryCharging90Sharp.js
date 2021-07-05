"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  fillOpacity: ".3",
  d: "M17 4h-3V2h-4v2H7v4h5.47L13 7v1h4V4z"
}), _react.default.createElement("path", {
  d: "M13 12.5h2L11 20v-5.5H9L12.47 8H7v14h10V8h-4v4.5z"
})), 'BatteryCharging90Sharp');

exports.default = _default;