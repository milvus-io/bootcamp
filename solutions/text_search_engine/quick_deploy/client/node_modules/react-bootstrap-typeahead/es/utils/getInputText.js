import getOptionLabel from './getOptionLabel';
import { head } from './nodash';

function getInputText(props) {
  var activeItem = props.activeItem,
      labelKey = props.labelKey,
      multiple = props.multiple,
      selected = props.selected,
      text = props.text;

  if (activeItem) {
    // Display the input value if the pagination item is active.
    return getOptionLabel(activeItem, labelKey);
  }

  var selectedItem = !multiple && !!selected.length && head(selected);

  if (selectedItem) {
    return getOptionLabel(selectedItem, labelKey);
  }

  return text;
}

export default getInputText;