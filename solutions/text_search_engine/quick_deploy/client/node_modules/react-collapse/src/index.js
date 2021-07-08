const {Collapse} = require('./Collapse');
const {UnmountClosed} = require('./UnmountClosed');


// Default export
module.exports = UnmountClosed;


// Extra "named exports"
UnmountClosed.Collapse = Collapse;
UnmountClosed.UnmountClosed = UnmountClosed;
