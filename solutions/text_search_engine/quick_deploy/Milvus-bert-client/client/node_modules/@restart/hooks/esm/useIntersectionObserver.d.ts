/**
 * Setup an [`IntersectionObserver`](https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver) on
 * a DOM Element.
 *
 * @param element The DOM element to observe
 * @param init IntersectionObserver options
 */
export default function useIntersectionObserver<TElement extends Element>(element: TElement | null | undefined, { threshold, root, rootMargin }?: IntersectionObserverInit): IntersectionObserverEntry[];
