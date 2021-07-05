"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M4 5h3v13H4zm14 0h3v13h-3zM8 18h9V5H8v13zm2-11h5v9h-5V7z"
}), _react.default.createElement("path", {
  d: "M10 7h5v9h-5z",
  opacity: ".3"
})), 'ViewArrayTwoTone');

exports.default = _default;