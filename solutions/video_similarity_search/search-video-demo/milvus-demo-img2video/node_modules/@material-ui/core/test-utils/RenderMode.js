"use strict";

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard");

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.RenderContext = RenderContext;
exports.useIsSsr = useIsSsr;

var _react = _interopRequireDefault(require("react"));

var PropTypes = _interopRequireWildcard(require("prop-types"));

var Context = _react.default.createContext();

if (process.env.NODE_ENV !== 'production') {
  Context.displayName = 'RenderContext';
}
/**
 * @ignore - internal component.
 */


function RenderContext(_ref) {
  var children = _ref.children;
  return _react.default.createElement(Context.Provider, {
    value: "render"
  }, children);
}

process.env.NODE_ENV !== "production" ? RenderContext.propTypes = {
  children: PropTypes.node.isRequired
} : void 0;

function useIsSsr() {
  return _react.default.useContext(Context) === 'render';
}