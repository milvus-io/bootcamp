import { Interface } from "readline";

export type TypeAnchorOrigin = {
  vertical: "top" | "bottom";
  horizontal: "left" | "center" | "right";
};

export type TypeSnackbarType = "error" | "warning" | "success" | "info";

export type TypeSnackbarConfig = {
  open: boolean;
  anchorOrigin: TypeAnchorOrigin;
  message: string;
  type: TypeSnackbarType;
};

export type TypeDialogType = {
  open: boolean;
  type: "notice" | "custom";
  bgcolor?: string;
  params: {
    title?: string;
    component?: React.ReactNode;
    confirm?: () => void;
    cancel?: () => void;
    confirmLabel?: string;
    cancelLabel?: string;
    confirmClass?: string;
    cancelClass?: string;
  };
};

export interface IGetImagePath {
  image_path: string;
}
export interface IImagePath {}

export type TypeDialogConfigs = {
  open: boolean;
  bgcolor?: string;
  type?: string;
  params?: {
    title?: string;
    component: JSX.Element;
    confirm: () => void;
    cancel?: () => void;
  };
};

export interface IImageUpload {
  Table: string;
  File: string;
}

export interface IVideoSearchResult {
  distance: number;
  image: string;
  object: string;
  time: string;
}
