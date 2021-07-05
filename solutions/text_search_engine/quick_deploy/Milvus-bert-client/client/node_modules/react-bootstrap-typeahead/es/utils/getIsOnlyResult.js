import getOptionProperty from './getOptionProperty';
import { head } from './nodash';

function getIsOnlyResult(props) {
  var allowNew = props.allowNew,
      highlightOnlyResult = props.highlightOnlyResult,
      results = props.results;

  if (!highlightOnlyResult || allowNew) {
    return false;
  }

  return results.length === 1 && !getOptionProperty(head(results), 'disabled');
}

export default getIsOnlyResult;