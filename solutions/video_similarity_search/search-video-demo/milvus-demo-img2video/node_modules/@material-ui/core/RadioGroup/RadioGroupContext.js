"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

/**
 * @ignore - internal component.
 */
var RadioGroupContext = _react.default.createContext();

if (process.env.NODE_ENV !== 'production') {
  RadioGroupContext.displayName = 'RadioGroupContext';
}

var _default = RadioGroupContext;
exports.default = _default;