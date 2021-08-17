import PropTypes from 'prop-types';
import React from 'react';
import { css, StyleSheet } from 'aphrodite/no-important';

const Spinner = props => {
  const classes = StyleSheet.create(styles(props));

  return (
    <div className={css(classes.spinner)}>
      <div className={css(classes.ripple)} />
    </div>
  );
};

Spinner.propTypes = {
  color: PropTypes.string,
  size: PropTypes.number,
};

const rippleKeyframes = {
  '0%': {
    top: '50%',
    left: '50%',
    width: 0,
    height: 0,
    opacity: 1,
  },
  '100%': {
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    opacity: 0,
  },
};

const styles = ({ color, size }) => ({
  spinner: {
    display: 'inline-block',
    position: 'relative',
    width: size,
    height: size,
  },
  ripple: {
    position: 'absolute',
    border: `4px solid ${color}`,
    opacity: 1,
    borderRadius: '50%',
    animationName: rippleKeyframes,
    animationDuration: '1s',
    animationTimingFunction: 'cubic-bezier(0, 0.2, 0.8, 1)',
    animationIterationCount: 'infinite',
  },
});

export default Spinner;
