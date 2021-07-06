import React from 'react';
import PropTypes from 'prop-types';
import {Collapse} from './Collapse';


export class UnmountClosed extends React.PureComponent {
  static propTypes = {
    isOpened: PropTypes.bool.isRequired,
    onWork: PropTypes.func,
    onRest: PropTypes.func
  };

  static defaultProps = {
    onWork: undefined,
    onRest: undefined
  };


  constructor(props) {
    super(props);
    this.state = {isResting: true, isOpened: props.isOpened, isInitialRender: true};
  }


  componentDidUpdate(prevProps) {
    const {isOpened} = this.props;
    if (prevProps.isOpened !== isOpened) {
      // eslint-disable-next-line react/no-did-update-set-state
      this.setState({isResting: false, isOpened, isInitialRender: false});
    }
  }


  onWork = ({isOpened, ...rest}) => {
    this.setState({isResting: false, isOpened});

    const {onWork} = this.props;
    if (onWork) {
      onWork({isOpened, ...rest});
    }
  };


  onRest = ({isOpened, ...rest}) => {
    this.setState({isResting: true, isOpened, isInitialRender: false});

    const {onRest} = this.props;
    if (onRest) {
      onRest({isOpened, ...rest});
    }
  };


  getInitialStyle = () => {
    const {isOpened, isInitialRender} = this.state;
    if (isInitialRender) {
      return isOpened
        ? {height: 'auto', overflow: 'initial'}
        : {height: '0px', overflow: 'hidden'};
    }

    return {height: '0px', overflow: 'hidden'};
  };


  render() {
    const {isResting, isOpened} = this.state;

    return isResting && !isOpened ? null : (
      <Collapse
        {...this.props}
        initialStyle={this.getInitialStyle()}
        onWork={this.onWork}
        onRest={this.onRest} />
    );
  }
}
