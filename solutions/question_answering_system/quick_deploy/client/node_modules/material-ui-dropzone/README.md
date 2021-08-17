# material-ui-dropzone
Material-UI-dropzone is a [React](https://github.com/facebook/react) component using [Material-UI](https://github.com/callemall/material-ui) and is based on the excellent [react-dropzone](https://github.com/react-dropzone/react-dropzone) library.

This components provide either a file-upload dropzone or a file-upload dropzone inside of a dialog.
The file-upload dropzone features some snazzy "File Allowed/Not Allowed" effects, previews and alerts.

## Installation
```sh
npm install --save material-ui-dropzone
```

## Screenshots
This is the component:
<br />
<img src="https://raw.githubusercontent.com/Yuvaleros/material-ui-dropzone/master/pics/demo_pic.jpg" width=400 height=300 /> 

When you drag a file onto the dropzone, you get a neat effect:
<br />
<img src="https://raw.githubusercontent.com/Yuvaleros/material-ui-dropzone/master/pics/demo_pic2.JPG" width=400 height=300 />  
<br />
<img src="https://raw.githubusercontent.com/Yuvaleros/material-ui-dropzone/master/pics/demo_pic5.JPG" width=400 height=300 />

And if you drop in a wrong type of file, you'll get yelled at:
<br />
<img src="https://raw.githubusercontent.com/Yuvaleros/material-ui-dropzone/master/pics/demo_pic4.JPG" width=400 height=300 />

## DropzoneArea Component

This components creates the dropzone, previews and snackbar notifications without a dialog

### Usage

```jsx
import React, {Component} from 'react'
import {DropzoneArea} from 'material-ui-dropzone'

class DropzoneAreaExample extends Component{
  constructor(props){
    super(props);
    this.state = {
      files: []
    };
  }
  handleChange(files){
    this.setState({
      files: files
    });
  }
  render(){
    return (
      <DropzoneArea 
        onChange={this.handleChange.bind(this)}
        />
    )  
  }
} 

export default DropzoneAreaExample;
```

### DropzoneArea Component Properties

| Name           |Type         |Default     |Description
|----------------|-------------|------------|--------------------------------
| acceptedFiles  | Array       |\['image/\*', 'video/\*', 'application/\*'] | A list of file mime types to accept. Does support wildcards.
| filesLimit    | Number       | 3           | Maximum number of files that can be loaded into the dropzone
| maxFileSize   | Number       | 3000000     | Maximum file size (in bytes) that the dropzone will accept
| dropzoneText  | String       | 'Drag and drop an image file here or click' | Text in dropzone
| dropzoneClass    | String | null             | Custom CSS class name for dropzone container.
| showPreviews  | Boolean | false       | Shows previews **BELOW** the Dropzone
| showPreviewsInDropzone| Boolean| true      | Shows preview **INSIDE** the dropzone
| useChipsForPreview| Boolean| false      | Uses deletable Material-ui Chip components to display file names
| previewChipProps| Object| {}      | Props to pass to the Material-ui Chip components
| previewGridClasses | Object | {}             | {container: string, item: string, image: string}. Custom CSS classNames for preview grid components.
| previewGridProps | Object      | {}          | {container: GridProps, item: GridProps}. Props to pass to the Material-ui Grid components.
| showAlerts    | Boolean | true             | shows styled snackbar alerts when files are dropped, deleted or rejected. 
| dropzoneParagraphClass    | String | null             | Custom CSS class name for text inside the container. 
| showFileNamesInPreview | Boolean | false | Shows file name under the image    
| showFileNames | Boolean | false | Shows file name under the dropzone image.
| initialFiles | Array | [] | A list of urls of already uploaded images. Please take care of CORS

### DropzoneArea Component Events

|Name            |Return Params|Description
|----------------|-------------|--------------------------------
|onChange        |files(array) | Fired when the user drops files into dropzone or deletes a file. Returns all the files currently loaded into the dropzone.
|onDrop          |files(array) | Fired when the user drops files into the dropzone. Returns the files dropped
|onDropRejected  |files(array) | Fired when a file is rejected because of wrong file type, size or goes beyond the filesLimit. Returns the files that were rejected
|onDelete        |file        | Fired when a file is deleted from the previews panel.

### DropzoneArea Componet Get Alert Messages

|Name                     |Params    |Return Params|Description|Default message
|-------------------------|----------|-------------|-----------|-----------
|getFileLimitExceedMessage|filesLimit|String       |Get alert message to display when files limit is exceed | Maximum allowed number of files exceeded. Only `${filesLimit}` allowed
|getFileAddedMessage      |fileName  |String       |Get alert message to display when a new file is added | File `${fileName}` successfully added.
|getFileRemovedMessage      |fileName  |String       |Get alert message to display when a file is removed | File `${fileName}` removed.
|getDropRejectMessage      |rejectedFile, acceptedFiles, maxFileSize  |String       |Get alert message to display when a file is removed | File `${rejectedFile.name}` was rejected..

## DropzoneDialog Component

This component provides the dropzone inside of a dialog. 

### Usage

```jsx
import React, { Component } from 'react'
import {DropzoneDialog} from 'material-ui-dropzone'
import Button from '@material-ui/core/Button';

export default class DropzoneDialogExample extends Component {
    constructor(props) {
        super(props);
        this.state = {
            open: false,
            files: []
        };
    }

    handleClose() {
        this.setState({
            open: false
        });
    }

    handleSave(files) {
        //Saving files to state for further use and closing Modal.
        this.setState({
            files: files, 
            open: false
        });
    }

    handleOpen() {
        this.setState({
            open: true,
        });
    }

    render() {
        return (
            <div>
                <Button onClick={this.handleOpen.bind(this)}>
                  Add Image
                </Button>
                <DropzoneDialog
                    open={this.state.open}
                    onSave={this.handleSave.bind(this)}
                    acceptedFiles={['image/jpeg', 'image/png', 'image/bmp']}
                    showPreviews={true}
                    maxFileSize={5000000}
                    onClose={this.handleClose.bind(this)}
                />
            </div>
        );
    }
}
```
### DropzoneDialog Component Properties

| Name           |Type         |Default     |Description
|----------------|-------------|------------|--------------------------------
| open           | Boolean     | false      | Required. Sets whether the dialog is open or closed 
| dialogTitle    | String      | true      | Sets dialog title.
| dialogProps    | Object      | {}         | Props to pass to the Material-ui Dialog component
| dropzoneText   | String      | true      | Sets dropzone text.
| cancelButtonText   | String      | true      | Sets cancel button text in dialog.
| submitButtonText   | String      | true      | Sets submit button text in dialog.
| acceptedFiles  | Array       |\['image/\*', 'video/\*', 'application/\*'] | A list of file mime types to accept. Does support wildcards.
| filesLimit    | Number       | 3           | Maximum number of files that can be loaded into the dropzone
| maxFileSize   | Number       | 3000000     | Maximum file size (in bytes) that the dropzone will accept
| showPreviews  | Boolean | false       | Shows previews **BELOW** the Dropzone
| showPreviewsInDropzone| Boolean| true      | Shows preview **INSIDE** the dropzone
| useChipsForPreview| Boolean| false      | Uses deletable Material-ui Chip components to display file names
| previewChipProps| Object| {}      | Props to pass to the Material-ui Chip components
| previewGridClasses | Object | {}             | {container: string, item: string, image: string}. Custom CSS classNames for preview grid components.
| previewGridProps | Object      | {}          | {container: GridProps, item: GridProps}. Props to pass to the Material-ui Grid components.
| showAlerts    | Boolean | true             | shows styled snackbar alerts when files are dropped, deleted or
| maxWidth      | String      | sm          | Sets dialog width. Width grows with the size of the screen.
| fullWidth    | Boolean     | true        | If true, the dialog stretches to maxWidth.


### DropzoneDialog Component Events

|Name            |Return Params|Description
|----------------|-------------|--------------------------------
| onSave         | files(array) | Fired when the user clicks the Submit button. 
| onClose        | event       | Fired when the modal is closed 
| onChange       |files(array) | Fired when the user drops files into dropzone **OR** deletes a file. Returns all the files currently loaded into the dropzone.
| onDrop         |files(array) | Fired when the user drops files into the dropzone. Returns the files dropped 
| onDropRejected |files(array) | Fired when a file is rejected because of wrong file type, size or goes beyond the filesLimit. Returns the files that were rejected
| onDelete       |file        | Fired when a file is deleted from the previews panel. 

## License
MIT
