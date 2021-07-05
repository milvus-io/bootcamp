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
  d: "M22 8V2L2 22h16V8z"
}), _react.default.createElement("path", {
  d: "M14 22V10L2 22h12zm6-12v8h2v-8h-2zm0 12h2v-2h-2v2z"
})), 'SignalCellularConnectedNoInternet2Bar');

exports.default = _default;