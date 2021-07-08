import useCustomEffect from './useCustomEffect';
import isEqual from "lodash-es/isEqual";
import useImmediateUpdateEffect from './useImmediateUpdateEffect';
import useEventCallback from './useEventCallback';

function isDepsEqual(_ref, _ref2) {
  var nextElement = _ref[0],
      nextConfig = _ref[1];
  var prevElement = _ref2[0],
      prevConfig = _ref2[1];
  return nextElement === prevElement && isEqual(nextConfig, prevConfig);
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
  var fn = useEventCallback(callback);
  useCustomEffect(function () {
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
    effectHook: useImmediateUpdateEffect
  });
}

export default useMutationObserver;