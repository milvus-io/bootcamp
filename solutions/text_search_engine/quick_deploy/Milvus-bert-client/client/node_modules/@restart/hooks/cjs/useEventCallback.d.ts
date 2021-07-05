export default function useEventCallback<TCallback extends (...args: any[]) => any>(fn?: TCallback | null): TCallback;
