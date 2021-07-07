import { DEFAULT_LABELKEY } from '../constants';
export default function getStringLabelKey(labelKey) {
  return typeof labelKey === 'string' ? labelKey : DEFAULT_LABELKEY;
}