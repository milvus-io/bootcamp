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
  d: "M23.64 7c-.45-.34-4.93-4-11.64-4C5.28 3 .81 6.66.36 7l10.08 12.56c.8 1 2.32 1 3.12 0L23.64 7z"
}), _react.default.createElement("path", {
  d: "M3.53 10.94l6.91 8.61c.8 1 2.32 1 3.12 0l6.91-8.61c-.43-.33-3.66-2.95-8.47-2.95s-8.04 2.62-8.47 2.95z"
})), 'NetworkWifiRounded');

exports.default = _default;