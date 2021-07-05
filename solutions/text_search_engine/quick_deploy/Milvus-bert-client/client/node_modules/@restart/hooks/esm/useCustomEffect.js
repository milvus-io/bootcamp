import { useRef, useEffect, useDebugValue } from 'react';
import useMounted from './useMounted';

function useCustomEffect(effect, dependencies, isEqualOrOptions) {
  var isMounted = useMounted();

  var _ref = typeof isEqualOrOptions === 'function' ? {
    isEqual: isEqualOrOptions
  } : isEqualOrOptions,
      isEqual = _ref.isEqual,
      _ref$effectHook = _ref.effectHook,
      effectHook = _ref$effectHook === void 0 ? useEffect : _ref$effectHook;

  var dependenciesRef = useRef();
  dependenciesRef.current = dependencies;
  var cleanupRef = useRef(null);
  effectHook(function () {
    // If the ref the is `null` it's either the first effect or the last effect
    // ran and was cleared, meaning _this_ update should run, b/c the equality
    // check failed on in the cleanup of the last effect.
    if (cleanupRef.current === null) {
      var cleanup = effect();

      cleanupRef.current = function () {
        if (isMounted() && isEqual(dependenciesRef.current, dependencies)) {
          return;
        }

        cleanupRef.current = null;
        if (cleanup) cleanup();
      };
    }

    return cleanupRef.current;
  });
  useDebugValue(effect);
}

export default useCustomEffect;