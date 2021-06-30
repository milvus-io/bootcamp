"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M7 5h14v2H7z"
}), _react.default.createElement("circle", {
  cx: "4",
  cy: "6",
  r: "1.5"
}), _react.default.createElement("path", {
  d: "M7 11h14v2H7zm0 6h14v2H7zm-3 2.5c.82 0 1.5-.68 1.5-1.5s-.67-1.5-1.5-1.5-1.5.68-1.5 1.5.68 1.5 1.5 1.5z"
}), _react.default.createElement("circle", {
  cx: "4",
  cy: "12",
  r: "1.5"
})), 'FormatListBulletedTwoTone');

exports.default = _default;