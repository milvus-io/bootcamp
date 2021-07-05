"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("circle", {
  cx: "7.2",
  cy: "14.4",
  r: "3.2"
}), _react.default.createElement("circle", {
  cx: "14.8",
  cy: "18",
  r: "2"
}), _react.default.createElement("circle", {
  cx: "15.2",
  cy: "8.8",
  r: "4.8"
})), 'BubbleChartRounded');

exports.default = _default;