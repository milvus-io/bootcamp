import * as React from "react";
import { DialogOptions } from "../Dialog";
export declare function isFocusTrap(element: Element): boolean;
export declare function useFocusTrap(dialogRef: React.RefObject<HTMLElement>, nestedDialogs: Array<React.RefObject<HTMLElement>>, options: DialogOptions): void;
