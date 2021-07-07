"use strict";

exports.__esModule = true;
exports.default = useGlobalListener;

var _useEventListener = _interopRequireDefault(require("./useEventListener"));

var _react = require("react");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Attaches an event handler outside directly to the `document`,
 * bypassing the react synthetic event system.
 *
 * ```ts
 * useGlobalListener('keydown', (event) => {
 *  console.log(event.key)
 * })
 * ```
 *
 * @param event The DOM event name
 * @param handler An event handler
 * @param capture Whether or not to listen during the capture event phase
 */
function useGlobalListener(event, handler, capture) {
  if (capture === void 0) {
    capture = false;
  }

  var documentTarget = (0, _react.useCallback)(function () {
    return document;
  }, []);
  return (0, _useEventListener.default)(documentTarget, event, handler, capture);
}