import { useEffect } from 'react';
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
  return useEffect(effect, []);
}

export default useMountEffect;