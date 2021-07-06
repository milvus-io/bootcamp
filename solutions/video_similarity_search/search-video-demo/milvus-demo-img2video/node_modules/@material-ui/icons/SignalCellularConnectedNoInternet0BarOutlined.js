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
  d: "M20 22h2v-2h-2v2zm0-12v8h2v-8h-2z"
})), 'SignalCellularConnectedNoInternet0BarOutlined');

exports.default = _default;