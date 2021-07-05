"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("circle", {
  cx: "7",
  cy: "14",
  r: "3"
}), _react.default.createElement("circle", {
  cx: "11",
  cy: "6",
  r: "3"
}), _react.default.createElement("circle", {
  cx: "16.6",
  cy: "17.6",
  r: "3"
})), 'ScatterPlot');

exports.default = _default;