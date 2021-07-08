import React from 'react';
import PropTypes from 'prop-types';


export class Collapse extends React.Component {
  static propTypes = {
    theme: PropTypes.shape({
      collapse: PropTypes.string,
      content: PropTypes.string
    }),
    isOpened: PropTypes.bool.isRequired,
    initialStyle: PropTypes.shape({
      height: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      overflow: PropTypes.string
    }),
    onRest: PropTypes.func,
    onWork: PropTypes.func,
    checkTimeout: PropTypes.number,
    children: PropTypes.node.isRequired
  };


  static defaultProps = {
    theme: {
      collapse: 'ReactCollapse--collapse',
      content: 'ReactCollapse--content'
    },
    initialStyle: undefined,
    onRest: undefined,
    onWork: undefined,
    checkTimeout: 50
  };


  timeout = undefined;

  container = undefined;

  content = undefined;


  constructor(props) {
    super(props);
    if (props.initialStyle) {
      this.initialStyle = props.initialStyle;
    } else {
      this.initialStyle = props.isOpened
        ? {height: 'auto', overflow: 'initial'}
        : {height: '0px', overflow: 'hidden'};
    }
  }


  componentDidMount() {
    this.onResize();
  }


  shouldComponentUpdate(nextProps) {
    const {theme, isOpened, children} = this.props;

    return children !== nextProps.children
      || isOpened !== nextProps.isOpened
      || Object.keys(theme).some(c => theme[c] !== nextProps.theme[c]);
  }


  getSnapshotBeforeUpdate() {
    if (!this.container || !this.content) {
      return null;
    }
    if (this.container.style.height === 'auto') {
      const {clientHeight: contentHeight} = this.content;
      this.container.style.height = `${contentHeight}px`;
    }
    return null;
  }


  componentDidUpdate() {
    this.onResize();
  }


  componentWillUnmount() {
    global.clearTimeout(this.timeout);
  }


  onResize = () => {
    global.clearTimeout(this.timeout);

    if (!this.container || !this.content) {
      return;
    }

    const {isOpened, checkTimeout} = this.props;
    const {clientHeight: containerHeight} = this.container;
    const {clientHeight: contentHeight} = this.content;

    const isFullyOpened = isOpened && contentHeight === containerHeight;
    const isFullyClosed = !isOpened && containerHeight === 0;

    if (isFullyOpened || isFullyClosed) {
      this.onRest({isFullyOpened, isFullyClosed, isOpened, containerHeight, contentHeight});
    } else {
      this.onWork({isFullyOpened, isFullyClosed, isOpened, containerHeight, contentHeight});
      this.timeout = setTimeout(() => this.onResize(), checkTimeout);
    }
  };


  onRest = ({isFullyOpened, isFullyClosed, isOpened, containerHeight, contentHeight}) => {
    if (!this.container || !this.content) {
      return;
    }

    const hasOpened = isOpened && this.container.style.height === `${contentHeight}px`;
    const hasClosed = !isOpened && this.container.style.height === '0px';

    if (hasOpened || hasClosed) {
      this.container.style.overflow = isOpened ? 'initial' : 'hidden';
      this.container.style.height = isOpened ? 'auto' : '0px';

      const {onRest} = this.props;
      if (onRest) {
        onRest({isFullyOpened, isFullyClosed, isOpened, containerHeight, contentHeight});
      }
    }
  };


  onWork = ({isFullyOpened, isFullyClosed, isOpened, containerHeight, contentHeight}) => {
    if (!this.container || !this.content) {
      return;
    }

    const isOpenining = isOpened && this.container.style.height === `${contentHeight}px`;
    const isClosing = !isOpened && this.container.style.height === '0px';

    if (isOpenining || isClosing) {
      // No need to do any work
      return;
    }

    this.container.style.overflow = 'hidden';
    this.container.style.height = isOpened ? `${contentHeight}px` : '0px';

    const {onWork} = this.props;
    if (onWork) {
      onWork({isFullyOpened, isFullyClosed, isOpened, containerHeight, contentHeight});
    }
  };


  onRefContainer = container => {
    this.container = container;
  };


  onRefContent = content => {
    this.content = content;
  };


  render() {
    const {theme, children} = this.props;
    return (
      <div ref={this.onRefContainer} className={theme.collapse} style={this.initialStyle}>
        <div ref={this.onRefContent} className={theme.content}>
          {children}
        </div>
      </div>
    );
  }
}
