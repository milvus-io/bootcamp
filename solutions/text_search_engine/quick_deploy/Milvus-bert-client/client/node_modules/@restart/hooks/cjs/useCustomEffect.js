"use strict";

exports.__esModule = true;
exports.default = void 0;

var _react = require("react");

var _useMounted = _interopRequireDefault(require("./useMounted"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function useCustomEffect(effect, dependencies, isEqualOrOptions) {
  var isMounted = (0, _useMounted.default)();

  var _ref = typeof isEqualOrOptions === 'function' ? {
    isEqual: isEqualOrOptions
  } : isEqualOrOptions,
      isEqual = _ref.isEqual,
      _ref$effectHook = _ref.effectHook,
      effectHook = _ref$effectHook === void 0 ? _react.useEffect : _ref$effectHook;

  var dependenciesRef = (0, _react.useRef)();
  dependenciesRef.current = dependencies;
  var cleanupRef = (0, _react.useRef)(null);
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
  (0, _react.useDebugValue)(effect);
}

var _default = useCustomEffect;
exports.default = _default;