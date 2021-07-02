import PropTypes from 'prop-types';
import React, { DragEvent as ReactDragEvent, DragEventHandler as ReactDragEventHandler } from 'react';

export type TDropEffects = 'copy' | 'move' | 'link' | 'none';

export interface IFileDropProps {
  className?: string;
  targetClassName?: string;
  draggingOverFrameClassName?: string;
  draggingOverTargetClassName?: string;
  frame?: HTMLElement | Document;
  onFrameDragEnter?: (event:DragEvent) => void;
  onFrameDragLeave?: (event:DragEvent) => void;
  onFrameDrop?: (event:DragEvent) => void;
  onDragOver?: ReactDragEventHandler<HTMLDivElement>;
  onDragLeave?: ReactDragEventHandler<HTMLDivElement>;
  onDrop?: (files:FileList | null, event:ReactDragEvent<HTMLDivElement>) => any;
  dropEffect?: TDropEffects;
}
export interface IFileDropState {
  draggingOverFrame: boolean;
  draggingOverTarget: boolean;
}

class FileDrop extends React.PureComponent<IFileDropProps, IFileDropState> {
  static defaultProps = {
    dropEffect: ('copy' as TDropEffects),
    frame: window ? window.document : undefined,
    className: 'file-drop',
    targetClassName: 'file-drop-target',
    draggingOverFrameClassName: 'file-drop-dragging-over-frame',
    draggingOverTargetClassName: 'file-drop-dragging-over-target',
  };

  static propTypes = {
    className: PropTypes.string,
    targetClassName: PropTypes.string,
    draggingOverFrameClassName: PropTypes.string,
    draggingOverTargetClassName: PropTypes.string,
    onDragOver: PropTypes.func,
    onDragLeave: PropTypes.func,
    onDrop: PropTypes.func,
    dropEffect: PropTypes.oneOf(['copy', 'move', 'link', 'none']),
    frame: (props:any, propName:any, componentName:any) => {
      const prop = props[propName];
      if (prop == null) {
        return new Error('Warning: Required prop `' + propName + '` was not specified in `' + componentName + '`');
      }
      if (prop !== document && prop !== window && !(prop instanceof HTMLElement)) {
        return new Error('Warning: Prop `' + propName + '` must be one of the following: document, HTMLElement!');
      }
    },
    onFrameDragEnter: PropTypes.func,
    onFrameDragLeave: PropTypes.func,
    onFrameDrop: PropTypes.func,
  };

  frameDragCounter: number;

  constructor(props:IFileDropProps) {
    super(props);
    this.frameDragCounter = 0;
    this.state = { draggingOverFrame: false, draggingOverTarget: false };
  }

  static isIE = () => (
    (window && ((window.navigator.userAgent.indexOf('MSIE') !== -1) || (window.navigator.appVersion.indexOf('Trident/') > 0)))
  )

  static eventHasFiles = (event:DragEvent | ReactDragEvent<HTMLElement>) => {
    // In most browsers this is an array, but in IE11 it's an Object :(
    let hasFiles = false;
    if (event.dataTransfer) {
      const types = event.dataTransfer.types;
      for (const keyOrIndex in types) {
        if (types[keyOrIndex] === 'Files') {
          hasFiles = true;
          break;
        }
      }
    }
    return hasFiles;
  }

  resetDragging = () => {
    this.frameDragCounter = 0;
    this.setState({ draggingOverFrame: false, draggingOverTarget: false });
  }

  handleWindowDragOverOrDrop = (event:DragEvent) => {
    // This prevents the browser from trying to load whatever file the user dropped on the window
    event.preventDefault();
  }

  handleFrameDrag = (event:DragEvent) => {
    // Only allow dragging of files
    if (!FileDrop.eventHasFiles(event)) return;

    // We are listening for events on the 'frame', so every time the user drags over any element in the frame's tree,
    // the event bubbles up to the frame. By keeping count of how many "dragenters" we get, we can tell if they are still
    // "draggingOverFrame" (b/c you get one "dragenter" initially, and one "dragenter"/one "dragleave" for every bubble)
    // This is far better than a "dragover" handler, which would be calling `setState` continuously.
    this.frameDragCounter += (event.type === 'dragenter' ? 1 : -1);

    if (this.frameDragCounter === 1) {
      this.setState({ draggingOverFrame: true });
      if (this.props.onFrameDragEnter) this.props.onFrameDragEnter(event);
      return;
    }

    if (this.frameDragCounter === 0) {
      this.setState({ draggingOverFrame: false });
      if (this.props.onFrameDragLeave) this.props.onFrameDragLeave(event);
      return;
    }
  }

  handleFrameDrop = (event:DragEvent) => {
    if (!this.state.draggingOverTarget) {
      this.resetDragging();
      if (this.props.onFrameDrop) this.props.onFrameDrop(event);
    }
  }

  handleDragOver: ReactDragEventHandler<HTMLDivElement> = (event) => {
    if (FileDrop.eventHasFiles(event)) {
      this.setState({ draggingOverTarget: true });
      if (!FileDrop.isIE() && this.props.dropEffect) event.dataTransfer.dropEffect = this.props.dropEffect;
      if (this.props.onDragOver) this.props.onDragOver(event);
    }
  }

  handleDragLeave: ReactDragEventHandler<HTMLDivElement> = (event) => {
    this.setState({ draggingOverTarget: false });
    if (this.props.onDragLeave) this.props.onDragLeave(event);
  }

  handleDrop: ReactDragEventHandler<HTMLDivElement> = (event) => {
    if (this.props.onDrop && FileDrop.eventHasFiles(event)) {
      const files = (event.dataTransfer) ? event.dataTransfer.files : null;
      this.props.onDrop(files, event);
    }
    this.resetDragging();
  }

  stopFrameListeners = (frame:IFileDropProps['frame']) => {
    if (frame) {
      frame.removeEventListener('dragenter', this.handleFrameDrag);
      frame.removeEventListener('dragleave', this.handleFrameDrag);
      frame.removeEventListener('drop', this.handleFrameDrop);
    }
  }

  startFrameListeners = (frame:IFileDropProps['frame']) => {
    if (frame) {
      frame.addEventListener('dragenter', this.handleFrameDrag);
      frame.addEventListener('dragleave', this.handleFrameDrag);
      frame.addEventListener('drop', this.handleFrameDrop);
    }
  }

  componentWillReceiveProps(nextProps:IFileDropProps) {
    if (nextProps.frame !== this.props.frame) {
      this.resetDragging();
      this.stopFrameListeners(this.props.frame);
      this.startFrameListeners(nextProps.frame);
    }
  }

  componentDidMount() {
    this.startFrameListeners(this.props.frame);
    this.resetDragging();
    window.addEventListener('dragover', this.handleWindowDragOverOrDrop);
    window.addEventListener('drop', this.handleWindowDragOverOrDrop);
  }

  componentWillUnmount() {
    this.stopFrameListeners(this.props.frame);
    window.removeEventListener('dragover', this.handleWindowDragOverOrDrop);
    window.removeEventListener('drop', this.handleWindowDragOverOrDrop);
  }

  render() {
    const {
      children,
      className,
      targetClassName,
      draggingOverFrameClassName,
      draggingOverTargetClassName } = this.props;
    const {
      draggingOverTarget,
      draggingOverFrame } = this.state;

    let fileDropTargetClassName = targetClassName;
    if (draggingOverFrame) fileDropTargetClassName += ` ${draggingOverFrameClassName}`;
    if (draggingOverTarget) fileDropTargetClassName += ` ${draggingOverTargetClassName}`;

    return (
      <div
        className={className}
        onDragOver={this.handleDragOver}
        onDragLeave={this.handleDragLeave}
        onDrop={this.handleDrop}
       >
        <div className={fileDropTargetClassName}>
          {children}
        </div>
      </div>
    );
  }
}

export default FileDrop;
