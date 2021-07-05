"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M15 7v9h-5V7h5m6-2h-3v13h3V5zm-4 0H8v13h9V5zM7 5H4v13h3V5z"
}), 'ViewArrayOutlined');

exports.default = _default;