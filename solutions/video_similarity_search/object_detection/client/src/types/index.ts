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

export interface IGetImagePath {
  image_path: string;
}
export interface IImagePath {}
