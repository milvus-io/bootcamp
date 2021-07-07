/**
 * Returns a controller object for setting a timeout that is properly cleaned up
 * once the component unmounts. New timeouts cancel and replace existing ones.
 */
export default function useTimeout(): {
    set: (fn: () => void, delayMs?: number) => void;
    clear: () => void;
};
