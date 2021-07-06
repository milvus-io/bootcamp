import canUseDOM from './canUseDOM';
import css from './css';
import listen from './listen';
export var TRANSITION_SUPPORTED = canUseDOM && 'ontransitionend' in window;
export function parseDuration(node) {
  var str = css(node, 'transitionDuration') || '';
  var mult = str.indexOf('ms') === -1 ? 1000 : 1;
  return parseFloat(str) * mult;
}

function triggerTransitionEnd(element) {
  var evt = document.createEvent('HTMLEvents');
  evt.initEvent('transitionend', true, true);
  element.dispatchEvent(evt);
}

export function emulateTransitionEnd(element, duration, padding) {
  if (padding === void 0) {
    padding = 5;
  }

  var called = false;
  var handle = setTimeout(function () {
    if (!called) triggerTransitionEnd(element);
  }, duration + padding);
  var remove = listen(element, 'transitionend', function () {
    called = true;
  }, {
    once: true
  });
  return function () {
    clearTimeout(handle);
    remove();
  };
}

function transitionEnd(element, handler, duration) {
  if (!TRANSITION_SUPPORTED) {
    return emulateTransitionEnd(element, 0, 0);
  }

  if (duration == null) duration = parseDuration(element) || 0;
  emulateTransitionEnd(element, duration);
  return listen(element, 'transitionend', handler);
}

export default transitionEnd;