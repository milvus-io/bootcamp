"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M4 10h12v2H4zm0-4h12v2H4zm0 8h8v2H4zm10 6l5-3-5-3z"
}), 'PlaylistPlayTwoTone');

exports.default = _default;