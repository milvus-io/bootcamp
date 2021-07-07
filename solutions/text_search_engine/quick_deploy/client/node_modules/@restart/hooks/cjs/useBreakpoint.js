"use strict";

exports.__esModule = true;
exports.createBreakpointHook = createBreakpointHook;
exports.default = void 0;

var _useMediaQuery = _interopRequireDefault(require("./useMediaQuery"));

var _react = require("react");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Create a responsive hook we a set of breakpoint names and widths.
 * You can use any valid css units as well as a numbers (for pixels).
 *
 * **NOTE:** The object key order is important! it's assumed to be in order from smallest to largest
 *
 * ```ts
 * const useBreakpoint = createBreakpointHook({
 *  xs: 0,
 *  sm: 576,
 *  md: 768,
 *  lg: 992,
 *  xl: 1200,
 * })
 * ```
 *
 * **Watch out!** using string values will sometimes construct media queries using css `calc()` which
 * is NOT supported in media queries by all browsers at the moment. use numbers for
 * the widest range of browser support.
 *
 * @param breakpointValues A object hash of names to breakpoint dimensions
 */
function createBreakpointHook(breakpointValues) {
  var names = Object.keys(breakpointValues);

  function and(query, next) {
    if (query === next) {
      return next;
    }

    return query ? query + " and " + next : next;
  }

  function getNext(breakpoint) {
    return names[Math.min(names.indexOf(breakpoint) + 1, names.length - 1)];
  }

  function getMaxQuery(breakpoint) {
    var next = getNext(breakpoint);
    var value = breakpointValues[next];
    if (typeof value === 'number') value = value - 0.2 + "px";else value = "calc(" + value + " - 0.2px)";
    return "(max-width: " + value + ")";
  }

  function getMinQuery(breakpoint) {
    var value = breakpointValues[breakpoint];

    if (typeof value === 'number') {
      value = value + "px";
    }

    return "(min-width: " + value + ")";
  }
  /**
   * Match a set of breakpoints
   *
   * ```tsx
   * const MidSizeOnly = () => {
   *   const isMid = useBreakpoint({ lg: 'down', sm: 'up' });
   *
   *   if (isMid) return <div>On a Reasonable sized Screen!</div>
   *   return null;
   * }
   * ```
   * @param breakpointMap An object map of breakpoints and directions, queries are constructed using "and" to join
   * breakpoints together
   */


  function useBreakpoint(breakpointOrMap, direction) {
    if (direction === void 0) {
      direction = true;
    }

    var breakpointMap;

    if (typeof breakpointOrMap === 'object') {
      breakpointMap = breakpointOrMap;
    } else {
      var _breakpointMap;

      breakpointMap = (_breakpointMap = {}, _breakpointMap[breakpointOrMap] = direction, _breakpointMap);
    }

    var query = (0, _react.useMemo)(function () {
      return Object.entries(breakpointMap).reduce(function (query, _ref) {
        var key = _ref[0],
            direction = _ref[1];

        if (direction === 'up' || direction === true) {
          query = and(query, getMinQuery(key));
        }

        if (direction === 'down' || direction === true) {
          query = and(query, getMaxQuery(key));
        }

        return query;
      }, '');
    }, [JSON.stringify(breakpointMap)]);
    return (0, _useMediaQuery.default)(query);
  }

  return useBreakpoint;
}

var useBreakpoint = createBreakpointHook({
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200
});
var _default = useBreakpoint;
exports.default = _default;