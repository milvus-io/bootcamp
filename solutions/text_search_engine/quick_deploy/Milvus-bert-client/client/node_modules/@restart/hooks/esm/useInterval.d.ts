/**
 * Creates a `setInterval` that is properly cleaned up when a component unmounted
 *
 * @param fn an function run on each interval
 * @param ms The milliseconds duration of the interval
 */
declare function useInterval(fn: () => void, ms: number): void;
/**
 * Creates a pausable `setInterval` that is properly cleaned up when a component unmounted
 *
 * @param fn an function run on each interval
 * @param ms The milliseconds duration of the interval
 * @param paused Whether or not the interval is currently running
 */
declare function useInterval(fn: () => void, ms: number, paused: boolean): void;
/**
 * Creates a pausable `setInterval` that is properly cleaned up when a component unmounted
 *
 * @param fn an function run on each interval
 * @param ms The milliseconds duration of the interval
 * @param paused Whether or not the interval is currently running
 * @param runImmediately Whether to run the function immediately on mount or unpause
 * rather than waiting for the first interval to elapse
 */
declare function useInterval(fn: () => void, ms: number, paused: boolean, runImmediately: boolean): void;
export default useInterval;
