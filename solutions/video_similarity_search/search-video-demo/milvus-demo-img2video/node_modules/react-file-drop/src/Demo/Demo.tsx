import React, { DragEvent as ReactDragEvent } from 'react';
import ReactDOM from 'react-dom';

import FileDrop from 'src/FileDrop/FileDrop';

import './Demo.css';

class Demo extends React.Component {
  // handleFrameDragEnter = (event:DragEvent) => {
  //   console.log('handleFrameDragEnter', event);
  // }
  //
  // handleFrameDragLeave = (event:DragEvent) => {
  //   console.log('handleFrameDragLeave', event);
  // }
  //
  // handleFrameDrop = (event:DragEvent) => {
  //   console.log('handleFrameDrop', event);
  // }
  //
  // handleDragOver:ReactDragEventHandler<HTMLDivElement> = (event) => {
  //   console.log('handleDragOver', event);
  // }
  //
  // handleDragLeave:ReactDragEventHandler<HTMLDivElement> = (event) => {
  //   console.log('handleDragLeave', event);
  // }

  handleDrop = (files:FileList, event:ReactDragEvent<HTMLDivElement>) => {
    console.log('handleDrop!', files, event);
  }

  render() {
    const styles = { border: '1px solid black', width: 600, color: 'black', padding: 20 };
    return (
      <div id="react-file-drop-demo" style={styles}>
        <FileDrop onDrop={this.handleDrop}>
          Drop some files here!
        </FileDrop>
      </div>
    );
  }
}

const dest = document.getElementById('root');
const content = <Demo />;
ReactDOM.render(content, dest);
