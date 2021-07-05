"use strict";

exports.__esModule = true;
exports.default = useMediaQuery;

var _useIsomorphicEffect = _interopRequireDefault(require("./useIsomorphicEffect"));

var _react = require("react");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var isBool = function isBool(a) {
  return typeof a === 'boolean';
};

var matchers = new Map();

var getMatcher = function getMatcher(query) {
  if (!query || typeof window == 'undefined') return undefined;
  var mql = matchers.get(query);

  if (!mql) {
    mql = window.matchMedia(query);
    mql.refCount = 0;
    matchers.set(mql.media, mql);
  }

  return mql;
};
/**
 * Match a media query and get updates as the match changes. The media string is
 * passed directly to `window.matchMedia` and run as a Layout Effect, so initial
 * matches are returned before the browser has a chance to paint.
 *
 * ```tsx
 * function Page() {
 *   const isWide = useMediaQuery('min-width: 1000px')
 *
 *   return isWide ? "very wide" : 'not so wide'
 * }
 * ```
 *
 * Media query lists are also reused globally, hook calls for the same query
 * will only create a matcher once under the hood.
 *
 * @param query A media query
 */


function useMediaQuery(query) {
  var mql = getMatcher(query);

  var _useState = (0, _react.useState)(function () {
    return mql ? mql.matches : false;
  }),
      matches = _useState[0],
      setMatches = _useState[1];

  (0, _useIsomorphicEffect.default)(function () {
    var mql = getMatcher(query);

    if (!mql) {
      return setMatches(false);
    }

    var handleChange = function handleChange() {
      setMatches(mql.matches);
    };

    mql.refCount++;
    mql.addListener(handleChange);
    handleChange();
    return function () {
      mql.removeListener(handleChange);
      mql.refCount--;

      if (mql.refCount <= 0) {
        matchers.delete(mql.media);
      }

      mql = undefined;
    };
  }, [query]);
  return matches;
}