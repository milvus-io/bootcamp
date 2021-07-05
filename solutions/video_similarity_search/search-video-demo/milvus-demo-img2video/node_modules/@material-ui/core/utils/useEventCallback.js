"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useEventCallback;

var _react = _interopRequireDefault(require("react"));

var useEnhancedEffect = typeof window !== 'undefined' ? _react.default.useLayoutEffect : _react.default.useEffect;
/**
 * https://github.com/facebook/react/issues/14099#issuecomment-440013892
 *
 * @param {function} fn
 */

function useEventCallback(fn) {
  var ref = _react.default.useRef(fn);

  useEnhancedEffect(function () {
    ref.current = fn;
  });
  return _react.default.useCallback(function () {
    return (0, ref.current).apply(void 0, arguments);
  }, []);
}