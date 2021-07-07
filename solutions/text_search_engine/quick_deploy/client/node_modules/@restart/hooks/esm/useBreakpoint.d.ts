export declare type BreakpointDirection = 'up' | 'down' | true;
export declare type BreakpointMap<TKey extends string> = Partial<Record<TKey, BreakpointDirection>>;
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
export declare function createBreakpointHook<TKey extends string>(breakpointValues: Record<TKey, string | number>): {
    (breakpointMap: Partial<Record<TKey, BreakpointDirection>>): boolean;
    (breakpoint: TKey, direction?: true | "up" | "down" | undefined): boolean;
};
export declare type DefaultBreakpoints = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export declare type DefaultBreakpointMap = BreakpointMap<DefaultBreakpoints>;
declare const useBreakpoint: {
    (breakpointMap: Partial<Record<DefaultBreakpoints, BreakpointDirection>>): boolean;
    (breakpoint: DefaultBreakpoints, direction?: true | "up" | "down" | undefined): boolean;
};
export default useBreakpoint;
