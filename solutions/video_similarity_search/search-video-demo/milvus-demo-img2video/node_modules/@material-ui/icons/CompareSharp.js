"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M10 3H3v18h7v2h2V1h-2v2zm0 15H5l5-6v6zM21 3h-7v2h5v13l-5-6v9h7V3z"
}), 'CompareSharp');

exports.default = _default;