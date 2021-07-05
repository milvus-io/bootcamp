"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M15.5 5H11l5 7-5 7h4.5l5-7z"
}), _react.default.createElement("path", {
  d: "M8.5 5H4l5 7-5 7h4.5l5-7z"
})), 'DoubleArrow');

exports.default = _default;