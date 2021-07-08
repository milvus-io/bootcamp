import isEqual from 'fast-deep-equal';
import getOptionProperty from './getOptionProperty';
import { isFunction, isString } from './nodash';
import stripDiacritics from './stripDiacritics';
import warn from './warn';

function isMatch(input, string, props) {
  var searchStr = input;
  var str = string;

  if (!props.caseSensitive) {
    searchStr = searchStr.toLowerCase();
    str = str.toLowerCase();
  }

  if (props.ignoreDiacritics) {
    searchStr = stripDiacritics(searchStr);
    str = stripDiacritics(str);
  }

  return str.indexOf(searchStr) !== -1;
}
/**
 * Default algorithm for filtering results.
 */


export default function defaultFilterBy(option, props) {
  var filterBy = props.filterBy,
      labelKey = props.labelKey,
      multiple = props.multiple,
      selected = props.selected,
      text = props.text; // Don't show selected options in the menu for the multi-select case.

  if (multiple && selected.some(function (o) {
    return isEqual(o, option);
  })) {
    return false;
  }

  if (isFunction(labelKey) && isMatch(text, labelKey(option), props)) {
    return true;
  }

  var fields = filterBy.slice();

  if (isString(labelKey)) {
    // Add the `labelKey` field to the list of fields if it isn't already there.
    if (fields.indexOf(labelKey) === -1) {
      fields.unshift(labelKey);
    }
  }

  if (isString(option)) {
    warn(fields.length <= 1, 'You cannot filter by properties when `option` is a string.');
    return isMatch(text, option, props);
  }

  return fields.some(function (field) {
    var value = getOptionProperty(option, field);

    if (!isString(value)) {
      warn(false, 'Fields passed to `filterBy` should have string values. Value will ' + 'be converted to a string; results may be unexpected.');
      value = String(value);
    }

    return isMatch(text, value, props);
  });
}