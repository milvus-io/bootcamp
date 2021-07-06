export declare type Listener = (this: HTMLElement, ev: TransitionEvent) => any;
export declare const TRANSITION_SUPPORTED: boolean;
export declare function parseDuration(node: HTMLElement): number;
export declare function emulateTransitionEnd(element: HTMLElement, duration: number, padding?: number): () => void;
declare function transitionEnd(element: HTMLElement, handler: Listener, duration?: number): () => void;
export default transitionEnd;
