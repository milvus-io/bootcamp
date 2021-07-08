"use strict";

exports.__esModule = true;
exports.default = useFocusManager;

var _react = require("react");

var _useMounted = _interopRequireDefault(require("./useMounted"));

var _useEventCallback = _interopRequireDefault(require("./useEventCallback"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * useFocusManager provides a way to track and manage focus as it moves around
 * a container element. An `onChange` is fired when focus enters or leaves the
 * element, but not when it moves around inside the element, similar to
 * `pointerenter` and `pointerleave` DOM events.
 *
 * ```ts
 * const [focused, setFocusState] = useState(false)
 *
 * const { onBlur, onFocus } = useFocusManager({
 *   onChange: nextFocused => setFocusState(nextFocused)
 * })
 *
 * return (
 *   <div tabIndex="-1" onFocus={onFocus} onBlur={onBlur}>
 *     {String(focused)}
 *     <input />
 *     <input />
 *
 *     <button>A button</button>
 *   </div>
 * ```
 *
 * @param opts Options
 * @returns FocusController a set of paired focus and blur event handlers
 */
function useFocusManager(opts) {
  var isMounted = (0, _useMounted.default)();
  var lastFocused = (0, _react.useRef)();
  var handle = (0, _react.useRef)();
  var willHandle = (0, _useEventCallback.default)(opts.willHandle);
  var didHandle = (0, _useEventCallback.default)(opts.didHandle);
  var onChange = (0, _useEventCallback.default)(opts.onChange);
  var isDisabled = (0, _useEventCallback.default)(opts.isDisabled);
  var handleFocusChange = (0, _react.useCallback)(function (focused, event) {
    if (event && event.persist) event.persist();
    if (willHandle && willHandle(focused, event) === false) return;
    clearTimeout(handle.current);
    handle.current = setTimeout(function () {
      if (focused !== lastFocused.current) {
        if (didHandle) didHandle(focused, event); // only fire a change when unmounted if its a blur

        if (isMounted() || !focused) {
          lastFocused.current = focused;
          onChange && onChange(focused, event);
        }
      }
    });
  }, [isMounted, willHandle, didHandle, onChange, lastFocused]);
  var handleBlur = (0, _react.useCallback)(function (event) {
    if (!isDisabled()) handleFocusChange(false, event);
  }, [handleFocusChange, isDisabled]);
  var handleFocus = (0, _react.useCallback)(function (event) {
    if (!isDisabled()) handleFocusChange(true, event);
  }, [handleFocusChange, isDisabled]);
  return {
    onBlur: handleBlur,
    onFocus: handleFocus
  };
}