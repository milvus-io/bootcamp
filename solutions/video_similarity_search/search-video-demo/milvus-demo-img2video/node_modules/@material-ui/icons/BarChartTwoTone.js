"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M5 9.2h3V19H5zM16.2 13H19v6h-2.8zm-5.6-8h2.8v14h-2.8z"
}), 'BarChartTwoTone');

exports.default = _default;