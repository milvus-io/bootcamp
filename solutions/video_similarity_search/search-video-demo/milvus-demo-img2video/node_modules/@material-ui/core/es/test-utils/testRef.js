import React from 'react';
import { assert } from 'chai';

function assertDOMNode(node) {
  // duck typing a DOM node
  assert.ok(node.nodeName);
}
/**
 * Utility method to make assertions about the ref on an element
 * @param {React.ReactElement} element - The element should have a component wrapped
 *                                       in withStyles as the root
 * @param {function} mount - Should be returnvalue of createMount
 * @param {function} onRef - Callback, first arg is the ref.
 *                           Assert that the ref is a DOM node by default
 */


export default function testRef(element, mount, onRef = assertDOMNode) {
  const ref = React.createRef();
  const wrapper = mount(React.createElement(React.Fragment, null, React.cloneElement(element, {
    ref
  })));
  onRef(ref.current, wrapper);
}