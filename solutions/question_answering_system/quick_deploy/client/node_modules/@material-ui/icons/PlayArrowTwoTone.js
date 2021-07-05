"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M10 8.64v6.72L15.27 12z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M8 19l11-7L8 5v14zm2-10.36L15.27 12 10 15.36V8.64z"
})), 'PlayArrowTwoTone');

exports.default = _default;