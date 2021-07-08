"use strict";

exports.__esModule = true;
exports.default = void 0;

var _react = require("react");

/**
 * Run's an effect on mount, and is cleaned up on unmount. Generally
 * useful for interop with non-react plugins or components
 *
 * ```ts
 *  useMountEffect(() => {
 *    const plugin = $.myPlugin(ref.current)
 *
 *    return () => {
 *      plugin.destroy()
 *    }
 *  })
 * ```
 * @param effect An effect to run on mount
 *
 *  @category effects
 */
function useMountEffect(effect) {
  return (0, _react.useEffect)(effect, []);
}

var _default = useMountEffect;
exports.default = _default;