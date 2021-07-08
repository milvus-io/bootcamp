import PropTypes from 'prop-types';
import { useContext } from 'react';
import DropdownContext from './DropdownContext';
/**
 * Wires up Dropdown toggle functinality, returning a set a props to attach
 * to the element that functions as the dropdown toggle (generally a button).
 */

export function useDropdownToggle() {
  var _useContext = useContext(DropdownContext),
      show = _useContext.show,
      toggle = _useContext.toggle,
      setToggle = _useContext.setToggle;

  return [{
    ref: setToggle,
    'aria-haspopup': true,
    'aria-expanded': !!show
  }, {
    show: show,
    toggle: toggle
  }];
}
var propTypes = {
  /**
   * A render prop that returns a Toggle element. The `props`
   * argument should spread through to **a component that can accept a ref**. Use
   * the `onToggle` argument to toggle the menu open or closed
   *
   * @type {Function ({
   *   show: boolean,
   *   toggle: (show: boolean) => void,
   *   props: {
   *     ref: (?HTMLElement) => void,
   *     aria-haspopup: true
   *     aria-expanded: boolean
   *   },
   * }) => React.Element}
   */
  children: PropTypes.func.isRequired
};

function DropdownToggle(_ref) {
  var children = _ref.children;

  var _useDropdownToggle = useDropdownToggle(),
      props = _useDropdownToggle[0],
      _useDropdownToggle$ = _useDropdownToggle[1],
      show = _useDropdownToggle$.show,
      toggle = _useDropdownToggle$.toggle;

  return children({
    show: show,
    toggle: toggle,
    props: props
  });
}

DropdownToggle.displayName = 'ReactOverlaysDropdownToggle';
DropdownToggle.propTypes = propTypes;
/** @component */

export default DropdownToggle;