import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";

/* eslint-disable jsx-a11y/aria-role */
import React from 'react';
import PropTypes from 'prop-types';
import clsx from 'clsx';
import KeyboardArrowLeft from '../internal/svg-icons/KeyboardArrowLeft';
import KeyboardArrowRight from '../internal/svg-icons/KeyboardArrowRight';
import withStyles from '../styles/withStyles';
import ButtonBase from '../ButtonBase';
export const styles = {
  root: {
    width: 40,
    flexShrink: 0
  },
  vertical: {
    width: '100%',
    height: 40,
    '& svg': {
      transform: 'rotate(90deg)'
    }
  }
};
/**
 * @ignore - internal component.
 */

var _ref = React.createElement(KeyboardArrowLeft, {
  fontSize: "small"
});

var _ref2 = React.createElement(KeyboardArrowRight, {
  fontSize: "small"
});

const TabScrollButton = React.forwardRef(function TabScrollButton(props, ref) {
  const {
    classes,
    className: classNameProp,
    direction,
    orientation,
    visible
  } = props,
        other = _objectWithoutPropertiesLoose(props, ["classes", "className", "direction", "orientation", "visible"]);

  const className = clsx(classes.root, classNameProp, orientation === 'vertical' && classes.vertical);

  if (!visible) {
    return React.createElement("div", {
      className: className
    });
  }

  return React.createElement(ButtonBase, _extends({
    component: "div",
    className: className,
    ref: ref,
    role: null,
    tabIndex: null
  }, other), direction === 'left' ? _ref : _ref2);
});
process.env.NODE_ENV !== "production" ? TabScrollButton.propTypes = {
  /**
   * Override or extend the styles applied to the component.
   * See [CSS API](#css) below for more details.
   */
  classes: PropTypes.object.isRequired,

  /**
   * @ignore
   */
  className: PropTypes.string,

  /**
   * Which direction should the button indicate?
   */
  direction: PropTypes.oneOf(['left', 'right']).isRequired,

  /**
   * The tabs orientation (layout flow direction).
   */
  orientation: PropTypes.oneOf(['horizontal', 'vertical']).isRequired,

  /**
   * Should the button be present or just consume space.
   */
  visible: PropTypes.bool.isRequired
} : void 0;
export default withStyles(styles, {
  name: 'PrivateTabScrollButton'
})(TabScrollButton);