import invariant from 'invariant';
import getStringLabelKey from './getStringLabelKey';
import { isFunction, isString } from './nodash';

/**
 * Retrieves the display string from an option. Options can be the string
 * themselves, or an object with a defined display string. Anything else throws
 * an error.
 */
function getOptionLabel(option, labelKey) {
  // Handle internally created options first.
  if (!isString(option) && (option.paginationOption || option.customOption)) {
    return option[getStringLabelKey(labelKey)];
  }

  var optionLabel;

  if (isFunction(labelKey)) {
    optionLabel = labelKey(option);
  } else if (isString(option)) {
    optionLabel = option;
  } else {
    // `option` is an object and `labelKey` is a string.
    optionLabel = option[labelKey];
  }

  !isString(optionLabel) ? process.env.NODE_ENV !== "production" ? invariant(false, 'One or more options does not have a valid label string. Check the ' + '`labelKey` prop to ensure that it matches the correct option key and ' + 'provides a string for filtering and display.') : invariant(false) : void 0;
  return optionLabel;
}

export default getOptionLabel;