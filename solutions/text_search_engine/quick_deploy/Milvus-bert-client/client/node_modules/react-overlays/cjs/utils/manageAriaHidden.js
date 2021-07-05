"use strict";

exports.__esModule = true;
exports.ariaHidden = ariaHidden;
exports.hideSiblings = hideSiblings;
exports.showSiblings = showSiblings;
var BLACKLIST = ['template', 'script', 'style'];

var isHidable = function isHidable(_ref) {
  var nodeType = _ref.nodeType,
      tagName = _ref.tagName;
  return nodeType === 1 && BLACKLIST.indexOf(tagName.toLowerCase()) === -1;
};

var siblings = function siblings(container, exclude, cb) {
  exclude = [].concat(exclude);
  [].forEach.call(container.children, function (node) {
    if (exclude.indexOf(node) === -1 && isHidable(node)) {
      cb(node);
    }
  });
};

function ariaHidden(show, node) {
  if (!node) return;

  if (show) {
    node.setAttribute('aria-hidden', 'true');
  } else {
    node.removeAttribute('aria-hidden');
  }
}

function hideSiblings(container, _ref2) {
  var dialog = _ref2.dialog,
      backdrop = _ref2.backdrop;
  siblings(container, [dialog, backdrop], function (node) {
    return ariaHidden(true, node);
  });
}

function showSiblings(container, _ref3) {
  var dialog = _ref3.dialog,
      backdrop = _ref3.backdrop;
  siblings(container, [dialog, backdrop], function (node) {
    return ariaHidden(false, node);
  });
}