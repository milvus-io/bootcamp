/// <reference types="react" />
declare type CallbackRef<T> = (ref: T | null) => void;
declare type Ref<T> = React.MutableRefObject<T> | CallbackRef<T>;
export declare function mergeRefs<T>(refA?: Ref<T> | null, refB?: Ref<T> | null): (value: T | null) => void;
/**
 * Create and returns a single callback ref composed from two other Refs.
 *
 * ```tsx
 * const Button = React.forwardRef((props, ref) => {
 *   const [element, attachRef] = useCallbackRef<HTMLButtonElement>();
 *   const mergedRef = useMergedRefs(ref, attachRef);
 *
 *   return <button ref={mergedRef} {...props}/>
 * })
 * ```
 *
 * @param refA A Callback or mutable Ref
 * @param refB A Callback or mutable Ref
 * @category refs
 */
declare function useMergedRefs<T>(refA?: Ref<T> | null, refB?: Ref<T> | null): (value: T | null) => void;
export default useMergedRefs;
