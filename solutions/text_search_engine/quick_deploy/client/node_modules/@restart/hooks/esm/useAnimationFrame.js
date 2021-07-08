import { useRef } from 'react';
import useMounted from './useMounted';
import useStableMemo from './useStableMemo';
import useWillUnmount from './useWillUnmount';

/**
 * Returns a controller object for requesting and cancelling an animation freame that is properly cleaned up
 * once the component unmounts. New requests cancel and replace existing ones.
 *
 * ```ts
 * const [style, setStyle] = useState({});
 * const animationFrame = useAnimationFrame();
 *
 * const handleMouseMove = (e) => {
 *   animationFrame.request(() => {
 *     setStyle({ top: e.clientY, left: e.clientY })
 *   })
 * }
 *
 * const handleMouseUp = () => {
 *   animationFrame.cancel()
 * }
 *
 * return (
 *   <div onMouseUp={handleMouseUp} onMouseMove={handleMouseMove}>
 *     <Ball style={style} />
 *   </div>
 * )
 * ```
 */
export default function useAnimationFrame() {
  var isMounted = useMounted();
  var handle = useRef();

  var cancel = function cancel() {
    if (handle.current != null) {
      cancelAnimationFrame(handle.current);
    }
  };

  useWillUnmount(cancel);
  return useStableMemo(function () {
    return {
      request: function request(cancelPrevious, fn) {
        if (!isMounted()) return;
        if (cancelPrevious) cancel();
        handle.current = requestAnimationFrame(fn || cancelPrevious);
      },
      cancel: cancel
    };
  }, []);
}