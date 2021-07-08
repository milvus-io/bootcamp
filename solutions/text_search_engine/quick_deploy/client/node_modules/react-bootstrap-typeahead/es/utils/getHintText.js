import getMatchBounds from './getMatchBounds';
import getOptionLabel from './getOptionLabel';

function getHintText(props) {
  var activeIndex = props.activeIndex,
      initialItem = props.initialItem,
      isFocused = props.isFocused,
      isMenuShown = props.isMenuShown,
      labelKey = props.labelKey,
      multiple = props.multiple,
      selected = props.selected,
      text = props.text; // Don't display a hint under the following conditions:

  if ( // No text entered.
  !text || // The input is not focused.
  !isFocused || // The menu is hidden.
  !isMenuShown || // No item in the menu.
  !initialItem || // The initial item is a custom option.
  initialItem.customOption || // One of the menu items is active.
  activeIndex > -1 || // There's already a selection in single-select mode.
  !!selected.length && !multiple) {
    return '';
  }

  var initialItemStr = getOptionLabel(initialItem, labelKey);
  var bounds = getMatchBounds(initialItemStr.toLowerCase(), text.toLowerCase());

  if (!(bounds && bounds.start === 0)) {
    return '';
  } // Text matching is case- and accent-insensitive, so to display the hint
  // correctly, splice the input string with the hint string.


  return text + initialItemStr.slice(bounds.end, initialItemStr.length);
}

export default getHintText;