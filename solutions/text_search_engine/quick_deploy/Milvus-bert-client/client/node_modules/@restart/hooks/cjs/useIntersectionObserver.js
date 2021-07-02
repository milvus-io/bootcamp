"use strict";

exports.__esModule = true;
exports.default = useIntersectionObserver;

var _react = require("react");

var _useStableMemo = _interopRequireDefault(require("./useStableMemo"));

var _useIsomorphicEffect = _interopRequireDefault(require("./useIsomorphicEffect"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Setup an [`IntersectionObserver`](https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver) on
 * a DOM Element.
 *
 * @param element The DOM element to observe
 * @param init IntersectionObserver options
 */
function useIntersectionObserver(element, _temp) {
  var _ref = _temp === void 0 ? {} : _temp,
      threshold = _ref.threshold,
      root = _ref.root,
      rootMargin = _ref.rootMargin;

  var _useState = (0, _react.useState)(null),
      entries = _useState[0],
      setEntry = _useState[1];

  var observer = (0, _useStableMemo.default)(function () {
    return typeof IntersectionObserver !== 'undefined' && new IntersectionObserver(function (entries) {
      return setEntry(entries);
    }, {
      threshold: threshold,
      root: root,
      rootMargin: rootMargin
    });
  }, [root, rootMargin, threshold && JSON.stringify(threshold)]);
  (0, _useIsomorphicEffect.default)(function () {
    if (!element || !observer) return;
    observer.observe(element);
    return function () {
      observer.unobserve(element);
    };
  }, [observer, element]);
  return entries || [];
}