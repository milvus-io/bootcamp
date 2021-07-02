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
  d: "M17 4h-3V2h-4v2H7v10.5h2L13 7v5.5h2l-1.07 2H17V4z"
}), _react.default.createElement("path", {
  d: "M11 20v-5.5H7V22h10v-7.5h-3.07L11 20z"
})), 'BatteryCharging30Sharp');

exports.default = _default;