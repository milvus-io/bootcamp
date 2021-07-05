import _extends from "@babel/runtime/helpers/esm/extends";
import React from 'react';
import PropTypes from 'prop-types';
import withStyles from '../styles/withStyles';
import Typography from '../Typography';
export var styles = {
  /* Styles applied to the root element. */
  root: {
    marginBottom: 12
  }
};
var DialogContentText = React.forwardRef(function DialogContentText(props, ref) {
  return React.createElement(Typography, _extends({
    component: "p",
    variant: "body1",
    color: "textSecondary",
    ref: ref
  }, props));
});
process.env.NODE_ENV !== "production" ? DialogContentText.propTypes = {
  /**
   * The content of the component.
   */
  children: PropTypes.node,

  /**
   * Override or extend the styles applied to the component.
   * See [CSS API](#css) below for more details.
   */
  classes: PropTypes.object.isRequired
} : void 0;
export default withStyles(styles, {
  name: 'MuiDialogContentText'
})(DialogContentText);