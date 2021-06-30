const key = {
  fullscreenEnabled: 0,
  fullscreenElement: 1,
  requestFullscreen: 2,
  exitFullscreen: 3,
  fullscreenchange: 4,
  fullscreenerror: 5,
};

const webkit = [
  'webkitFullscreenEnabled',
  'webkitFullscreenElement',
  'webkitRequestFullscreen',
  'webkitExitFullscreen',
  'webkitfullscreenchange',
  'webkitfullscreenerror',
];

const moz = [
  'mozFullScreenEnabled',
  'mozFullScreenElement',
  'mozRequestFullScreen',
  'mozCancelFullScreen',
  'mozfullscreenchange',
  'mozfullscreenerror',
];

const ms = [
  'msFullscreenEnabled',
  'msFullscreenElement',
  'msRequestFullscreen',
  'msExitFullscreen',
  'MSFullscreenChange',
  'MSFullscreenError',
];

// so it doesn't throw if no window or document
const document = typeof window !== 'undefined' && typeof window.document !== 'undefined' ? window.document : {};

const vendor = (
  ('fullscreenEnabled' in document && Object.keys(key)) ||
  (webkit[0] in document && webkit) ||
  (moz[0] in document && moz) ||
  (ms[0] in document && ms) ||
  []
);

export default {
  requestFullscreen: element => element[vendor[key.requestFullscreen]](),
  requestFullscreenFunction: element => element[vendor[key.requestFullscreen]],
  get exitFullscreen() { return document[vendor[key.exitFullscreen]].bind(document); },
  addEventListener: (type, handler, options) => document.addEventListener(vendor[key[type]], handler, options),
  removeEventListener: (type, handler, options) => document.removeEventListener(vendor[key[type]], handler, options),
  get fullscreenEnabled() { return Boolean(document[vendor[key.fullscreenEnabled]]); },
  set fullscreenEnabled(val) {},
  get fullscreenElement() { return document[vendor[key.fullscreenElement]]; },
  set fullscreenElement(val) {},
  get onfullscreenchange() { return document[`on${vendor[key.fullscreenchange]}`.toLowerCase()]; },
  set onfullscreenchange(handler) { return document[`on${vendor[key.fullscreenchange]}`.toLowerCase()] = handler; },
  get onfullscreenerror() { return document[`on${vendor[key.fullscreenerror]}`.toLowerCase()]; },
  set onfullscreenerror(handler) { return document[`on${vendor[key.fullscreenerror]}`.toLowerCase()] = handler; },
};
