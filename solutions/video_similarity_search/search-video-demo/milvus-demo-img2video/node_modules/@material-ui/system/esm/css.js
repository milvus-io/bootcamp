import _toConsumableArray from "@babel/runtime/helpers/esm/toConsumableArray";
import _extends from "@babel/runtime/helpers/esm/extends";
import PropTypes from 'prop-types';
import merge from './merge';

function omit(input, fields) {
  var output = {};
  Object.keys(input).forEach(function (prop) {
    if (fields.indexOf(prop) === -1) {
      output[prop] = input[prop];
    }
  });
  return output;
}

function css(styleFunction) {
  var newStyleFunction = function newStyleFunction(props) {
    var output = styleFunction(props);

    if (props.css) {
      return _extends({}, merge(output, styleFunction(_extends({
        theme: props.theme
      }, props.css))), {}, omit(props.css, [styleFunction.filterProps]));
    }

    return output;
  };

  newStyleFunction.propTypes = process.env.NODE_ENV !== 'production' ? _extends({}, styleFunction.propTypes, {
    css: PropTypes.object
  }) : {};
  newStyleFunction.filterProps = ['css'].concat(_toConsumableArray(styleFunction.filterProps));
  return newStyleFunction;
}

export default css;