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
  d: "M22 8V2L2 22h16V8h4z"
}), _react.default.createElement("path", {
  d: "M18 22V6L2 22h16zm2-12v8h2v-8h-2zm0 12h2v-2h-2v2z"
})), 'SignalCellularConnectedNoInternet3BarSharp');

exports.default = _default;