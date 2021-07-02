import { useState } from 'react';
import useStableMemo from './useStableMemo';
import useEffect from './useIsomorphicEffect';
/**
 * Setup an [`IntersectionObserver`](https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver) on
 * a DOM Element.
 *
 * @param element The DOM element to observe
 * @param init IntersectionObserver options
 */

export default function useIntersectionObserver(element, _temp) {
  var _ref = _temp === void 0 ? {} : _temp,
      threshold = _ref.threshold,
      root = _ref.root,
      rootMargin = _ref.rootMargin;

  var _useState = useState(null),
      entries = _useState[0],
      setEntry = _useState[1];

  var observer = useStableMemo(function () {
    return typeof IntersectionObserver !== 'undefined' && new IntersectionObserver(function (entries) {
      return setEntry(entries);
    }, {
      threshold: threshold,
      root: root,
      rootMargin: rootMargin
    });
  }, [root, rootMargin, threshold && JSON.stringify(threshold)]);
  useEffect(function () {
    if (!element || !observer) return;
    observer.observe(element);
    return function () {
      observer.unobserve(element);
    };
  }, [observer, element]);
  return entries || [];
}