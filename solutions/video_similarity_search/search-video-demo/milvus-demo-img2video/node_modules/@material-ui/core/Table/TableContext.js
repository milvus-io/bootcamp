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
var TableContext = _react.default.createContext();

if (process.env.NODE_ENV !== 'production') {
  TableContext.displayName = 'TableContext';
}

var _default = TableContext;
exports.default = _default;