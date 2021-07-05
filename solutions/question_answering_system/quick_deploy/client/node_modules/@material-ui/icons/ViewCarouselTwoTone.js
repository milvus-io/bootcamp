"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M18 6h4v11h-4zM7 19h10V4H7v15zM9 6h6v11H9V6zM2 6h4v11H2z"
}), _react.default.createElement("path", {
  d: "M9 6h6v11H9z",
  opacity: ".3"
})), 'ViewCarouselTwoTone');

exports.default = _default;