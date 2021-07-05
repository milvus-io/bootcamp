# react-file-drop
React component for Gmail or Facebook -like drag and drop file uploader. Drag files anywhere onto the window (or user defined "frame" prop)! Very extensible, provides many hooks so you can use it to develop any custom behavior that you desire.

## V2 is out! See the [changelog](https://github.com/sarink/react-file-drop/blob/master/CHANGELOG.md) before upgrading

## Demo/Example
http://sarink.github.io/react-file-drop/dist/Demo - A very simple demo with example code and sample CSS

## By default, there are no styles! You must include some CSS if you want to see anything!
You can grab the [demo CSS](https://raw.githubusercontent.com/sarink/react-file-drop/master/src/Demo/Demo.css) to get started

## Browser support
✅ Chrome <br/>
✅ Firefox <br/>
✅ Safari <br/>
✅ IE 11 <br/>
✅ IE Edge <br/>

## Typescript?
Yup! (For typing event handlers, use the native DragEvent for frame handlers, and the React lib's DragEvent for others)

## Why?
I wanted that behavior like facebook, gmail, etc. have where a part of the page highlights immediately when you start dragging a file anywhere on the window. I couldn't find any React component that already did this, so, I made one.

## Installation
``npm install react-file-drop``


## Usage
``import FileDrop from 'react-file-drop``

## How it works
First, you define the ``frame`` prop (default is the ``document``), whenever the user begins dragging file(s) anywhere over this frame, the ``<div class="file-drop-target">`` will be inserted into the DOM.  
Next, define an ``onDrop`` prop, whenever a user drops their files onto the target, this callback will be triggered.  
Lastly, you'll need to style it. Check out the Styling section below for details.

## Props
##### onDrop - function(files, event)
Callback when the user drops files onto the target

##### onDragOver - function(event)
Callback when the user is dragging over the target. Also adds the ``file-drop-dragging-over-target`` class to the ``file-drop-target``

##### onDragLeave - function(event)
Callback when the user leaves the target. Removes the ``file-drop-dragging-over-target`` class from the ``file-drop-target``

##### dropEffect - String "copy" || "move" || "link" || "none" (default: "copy")
Learn more about [HTML5 dropEffects](https://developer.mozilla.org/en-US/docs/Web/API/DataTransfer#dropEffect.28.29). Not available in IE :(

##### frame - document || HTMLElement  (default: document)
This is the "scope" or frame that the user must drag some file(s) over to kick things off.

##### onFrameDragEnter - function(event)
Callback when the user begins dragging over the ``frame``

##### onFrameDragLeave - function(event)
Callback when the user stops dragging over the ``frame``

##### onFrameDrop - function(event)
Callback when the user drops files *anywhere* over the ``frame``

##### className - string (default: file-drop)
Class given to the outer container div.

##### targetClassName - string (default: file-drop-target)
Class given to the target div.

##### draggingOverFrameClassName - string (default: file-drop-dragging-over-frame)
Class given to the target div when file is being dragged over frame.

##### draggingOverTargetClassName - string (default: file-drop-dragging-over-target)
Class given to the target div when file is being dragged over target.

## Styling
By default, the component comes with no styles. You can grab the [demo CSS](https://raw.githubusercontent.com/sarink/react-file-drop/master/src/Demo/Demo.css) to get you started.

##### .file-drop
The outer container element

##### .file-drop > .file-drop-target
This is the target the user has to drag their files to. It will be inserted into the DOM whenever the user starts dragging over the frame.

##### .file-drop > .file-drop-target.file-drop-dragging-over-frame
The ``file-drop-dragging-over-frame`` class will be added to the ``file-drop-target`` whenever the user begins dragging a file over the ``frame``, and it will be removed when they leave

##### .file-drop > .file-drop-target.file-drop-dragging-over-target
The ``file-drop-dragging-over-target`` class will be added to the ``file-drop-target`` whenever the user begins dragging a file over the ``file-drop-target`` div, and it will be removed when they leave

For custom class names you can use the following props:

* className
* targetClassName
* draggingOverFrameClassName
* draggingOverTargetClassName

## Contributing
Your PRs are welcome! To run the app locally:

Run `docker-compose up` - this will launch a node docker container, and execute `npm run start:dev`, which will launch the demo app at `http://localhost:3003` (you can change the port number inside `docker-compose.yml`)

Once the docker container is running and started...

If you made changes to the Demo app, run `docker-compose exec frontend npm run build:demo` before you push (this will output new files in `dist/Demo`)

If you made changes to the main FileDrop, run `docker-compose exec frontend npm run build:filedrop` before you push (this will output new files in `dist/FileDrop`)

Note: This repo uses `tslint`, you can run `docker-compose exec frontend npm run lint:watch` (or configure your IDE to read the `tslint.json` file)

If you're not into docker, you can also just execute `PORT=3003 npm run start:dev` (you can use whatever port number is your favorite, but `webpack.config.js` expects `process.env.PORT` to be defined) and leave off the `docker-compose exec frontend` prefix from the rest of the commands
