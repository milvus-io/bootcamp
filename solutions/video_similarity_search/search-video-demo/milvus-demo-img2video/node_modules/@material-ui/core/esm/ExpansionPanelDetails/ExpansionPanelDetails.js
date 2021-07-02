import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutProperties from "@babel/runtime/helpers/esm/objectWithoutProperties";
import React from 'react';
import PropTypes from 'prop-types';
import clsx from 'clsx';
import withStyles from '../styles/withStyles';
export var styles = {
  /* Styles applied to the root element. */
  root: {
    display: 'flex',
    padding: '8px 24px 24px'
  }
};
var ExpansionPanelDetails = React.forwardRef(function ExpansionPanelDetails(props, ref) {
  var classes = props.classes,
      className = props.className,
      other = _objectWithoutProperties(props, ["classes", "className"]);

  return React.createElement("div", _extends({
    className: clsx(classes.root, className),
    ref: ref
  }, other));
});
process.env.NODE_ENV !== "production" ? ExpansionPanelDetails.propTypes = {
  /**
   * The content of the expansion panel details.
   */
  children: PropTypes.node.isRequired,

  /**
   * Override or extend the styles applied to the component.
   * See [CSS API](#css) below for more details.
   */
  classes: PropTypes.object.isRequired,

  /**
   * @ignore
   */
  className: PropTypes.string
} : void 0;
export default withStyles(styles, {
  name: 'MuiExpansionPanelDetails'
})(ExpansionPanelDetails);