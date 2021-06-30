"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("circle", {
  cx: "16",
  cy: "17",
  r: "1",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M3 10h12v2H3zM3 14h8v2H3zM3 6h12v2H3z"
}), _react.default.createElement("path", {
  d: "M17 14.18c-.31-.11-.65-.18-1-.18-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3V8h3V6h-5v8.18zM16 18c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1z"
})), 'QueueMusicTwoTone');

exports.default = _default;