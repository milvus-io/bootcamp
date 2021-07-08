import { isString } from './nodash';
export default function getOptionProperty(option, key) {
  if (isString(option)) {
    return undefined;
  }

  return option[key];
}