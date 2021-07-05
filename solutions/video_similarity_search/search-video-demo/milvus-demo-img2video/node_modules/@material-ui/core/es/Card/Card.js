import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
import React from 'react';
import PropTypes from 'prop-types';
import clsx from 'clsx';
import Paper from '../Paper';
import withStyles from '../styles/withStyles';
export const styles = {
  /* Styles applied to the root element. */
  root: {
    overflow: 'hidden'
  }
};
const Card = React.forwardRef(function Card(props, ref) {
  const {
    classes,
    className,
    raised = false
  } = props,
        other = _objectWithoutPropertiesLoose(props, ["classes", "className", "raised"]);

  return React.createElement(Paper, _extends({
    className: clsx(classes.root, className),
    elevation: raised ? 8 : 1,
    ref: ref
  }, other));
});
process.env.NODE_ENV !== "production" ? Card.propTypes = {
  /**
   * The content of the component.
   */
  children: PropTypes.node,

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
   * If `true`, the card will use raised styling.
   */
  raised: PropTypes.bool
} : void 0;
export default withStyles(styles, {
  name: 'MuiCard'
})(Card);