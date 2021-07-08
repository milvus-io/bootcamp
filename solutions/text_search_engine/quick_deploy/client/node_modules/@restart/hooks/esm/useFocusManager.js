import { useCallback, useRef } from 'react';
import useMounted from './useMounted';
import useEventCallback from './useEventCallback';

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
export default function useFocusManager(opts) {
  var isMounted = useMounted();
  var lastFocused = useRef();
  var handle = useRef();
  var willHandle = useEventCallback(opts.willHandle);
  var didHandle = useEventCallback(opts.didHandle);
  var onChange = useEventCallback(opts.onChange);
  var isDisabled = useEventCallback(opts.isDisabled);
  var handleFocusChange = useCallback(function (focused, event) {
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
  var handleBlur = useCallback(function (event) {
    if (!isDisabled()) handleFocusChange(false, event);
  }, [handleFocusChange, isDisabled]);
  var handleFocus = useCallback(function (event) {
    if (!isDisabled()) handleFocusChange(true, event);
  }, [handleFocusChange, isDisabled]);
  return {
    onBlur: handleBlur,
    onFocus: handleFocus
  };
}