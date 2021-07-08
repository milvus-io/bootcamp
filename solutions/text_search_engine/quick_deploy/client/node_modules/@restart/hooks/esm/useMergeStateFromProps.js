import useMergeState from './useMergeState';
export default function useMergeStateFromProps(props, gDSFP, initialState) {
  var _useMergeState = useMergeState(initialState),
      state = _useMergeState[0],
      setState = _useMergeState[1];

  var nextState = gDSFP(props, state);
  if (nextState !== null) setState(nextState);
  return [state, setState];
}