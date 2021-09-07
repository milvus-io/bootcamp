import React from "react";
import Visibility from "@material-ui/icons/Visibility";
import VisibilityOff from "@material-ui/icons/VisibilityOff";
import LockIcon from "@material-ui/icons/Lock";
import AccountCircle from "@material-ui/icons/AccountCircle";
import ErrorIcon from "@material-ui/icons/Error";
import ArrowDropDownIcon from "@material-ui/icons/ArrowDropDown";

const icons = {
  visible: (props = {}) => <Visibility {...props} />,
  invisible: (props = {}) => <VisibilityOff {...props} />,
  lock: (props = {}) => <LockIcon {...props} />,
  account: (props = {}) => <AccountCircle {...props} />,
  error: (props = {}) => <ErrorIcon {...props} />,
  downArrow: (props = {}) => <ArrowDropDownIcon {...props} />,
};

export default icons;
