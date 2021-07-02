"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M3.84 5.49l7.29 12.96c.38.68 1.36.68 1.74 0l7.29-12.96c.38-.67-.11-1.49-.87-1.49H4.71c-.76 0-1.25.82-.87 1.49zM6.38 6h11.25L12 16 6.38 6z"
}), 'DetailsRounded');

exports.default = _default;