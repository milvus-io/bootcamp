// @flow
type WrapperFn = (...arg: any[]) => AnimationFrameID;
type CancelFn = {|
  cancel: () => void,
|};
type ResultFn = WrapperFn & CancelFn;

export default (fn: Function): ResultFn => {
  let lastArgs: any[] = [];
  let frameId: ?AnimationFrameID = null;

  const wrapperFn: WrapperFn = (...args: any): AnimationFrameID => {
    // Always capture the latest value
    lastArgs = args;

    // There is already a frame queued
    if (frameId) {
      return frameId;
    }

    // Schedule a new frame
    frameId = requestAnimationFrame(() => {
      frameId = null;
      fn(...lastArgs);
    });

    return frameId;
  };

  // Adding cancel property to result function
  wrapperFn.cancel = () => {
    if (!frameId) {
      return;
    }

    cancelAnimationFrame(frameId);
    frameId = null;
  };

  const resultFn: ResultFn = (wrapperFn: any);

  return resultFn;
};