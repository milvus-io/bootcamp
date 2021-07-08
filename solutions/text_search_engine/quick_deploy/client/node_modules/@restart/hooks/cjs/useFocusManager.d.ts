/// <reference types="react" />
export interface FocusManagerOptions {
    /**
     * A callback fired when focus shifts. returning `false` will prevent
     * handling the focus event
     */
    willHandle?(focused: boolean, event: React.FocusEvent): boolean | void;
    /**
     * A callback fired after focus is handled but before onChange is called
     */
    didHandle?(focused: boolean, event: React.FocusEvent): void;
    /**
     * A callback fired after focus has changed
     */
    onChange?(focused: boolean, event: React.FocusEvent): void;
    /**
     * When true, the event handlers will not report focus changes
     */
    isDisabled: () => boolean;
}
/**
 * useFocusManager provides a way to track and manage focus as it moves around
 * a container element. An `onChange` is fired when focus enters or leaves the
 * element, but not when it moves around inside the element, similar to
 * `pointerenter` and `pointerleave` DOM events.
 *
 * ```ts
 * const [focused, setFocusState] = useState(false)
 *
 * const { onBlur, onFocus } = useFocusManager({
 *   onChange: nextFocused => setFocusState(nextFocused)
 * })
 *
 * return (
 *   <div tabIndex="-1" onFocus={onFocus} onBlur={onBlur}>
 *     {String(focused)}
 *     <input />
 *     <input />
 *
 *     <button>A button</button>
 *   </div>
 * ```
 *
 * @param opts Options
 * @returns FocusController a set of paired focus and blur event handlers
 */
export default function useFocusManager(opts: FocusManagerOptions): {
    onBlur: (event: any) => void;
    onFocus: (event: any) => void;
};
