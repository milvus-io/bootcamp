"use strict";

exports.__esModule = true;
exports.default = void 0;

var _useCustomEffect = _interopRequireDefault(require("./useCustomEffect"));

var _isEqual = _interopRequireDefault(require("lodash/isEqual"));

var _useImmediateUpdateEffect = _interopRequireDefault(require("./useImmediateUpdateEffect"));

var _useEventCallback = _interopRequireDefault(require("./useEventCallback"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function isDepsEqual(_ref, _ref2) {
  var nextElement = _ref[0],
      nextConfig = _ref[1];
  var prevElement = _ref2[0],
      prevConfig = _ref2[1];
  return nextElement === prevElement && (0, _isEqual.default)(nextConfig, prevConfig);
}
/**
 * Observe mutations on a DOM node or tree of DOM nodes.
 * Depends on the `MutationObserver` api.
 *
 * ```ts
 * const [element, attachRef] = useCallbackRef(null);
 *
 * useMutationObserver(element, { subtree: true }, (records) => {
 *
 * });
 *
 * return (
 *   <div ref={attachRef} />
 * )
 * ```
 *
 * @param element The DOM element to observe
 * @param config The observer configuration
 * @param callback A callback fired when a mutation occurs
 */


function useMutationObserver(element, config, callback) {
  var fn = (0, _useEventCallback.default)(callback);
  (0, _useCustomEffect.default)(function () {
    if (!element) return; // The behavior around reusing mutation observers is confusing
    // observing again _should_ disable the last listener but doesn't
    // seem to always be the case, maybe just in JSDOM? In any case the cost
    // to redeclaring it is gonna be fairly low anyway, so make it simple

    var observer = new MutationObserver(fn);
    observer.observe(element, config);
    return function () {
      observer.disconnect();
    };
  }, [element, config], {
    isEqual: isDepsEqual,
    // Intentionally done in render, otherwise observer will miss any
    // changes made to the DOM during this update
    effectHook: _useImmediateUpdateEffect.default
  });
}

var _default = useMutationObserver;
exports.default = _default;