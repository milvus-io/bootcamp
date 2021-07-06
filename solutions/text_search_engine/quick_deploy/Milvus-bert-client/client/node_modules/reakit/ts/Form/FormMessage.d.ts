/// <reference types="react" />
import { PropsWithAs } from "reakit-utils/types";
import { BoxOptions, BoxHTMLProps } from "../Box/Box";
import { unstable_FormStateReturn } from "./FormState";
import { DeepPath } from "./__utils/types";
export declare type unstable_FormMessageOptions<V, P extends DeepPath<V, P>> = BoxOptions & Pick<unstable_FormStateReturn<V>, "baseId" | "touched" | "errors" | "messages"> & {
    /**
     * FormInput's name as in form values.
     */
    name: P;
};
export declare type unstable_FormMessageHTMLProps = BoxHTMLProps;
export declare type unstable_FormMessageProps<V, P extends DeepPath<V, P>> = unstable_FormMessageOptions<V, P> & unstable_FormMessageHTMLProps;
export declare const unstable_useFormMessage: <V, P extends DeepPath<V, P>>(options: unstable_FormMessageOptions<V, P>, htmlProps?: BoxHTMLProps | undefined) => BoxHTMLProps;
export declare const unstable_FormMessage: <V, P extends DeepPath<V, P>, T extends import("react").ElementType<any> = "div">(props: PropsWithAs<unstable_FormMessageOptions<V, P>, T>) => JSX.Element;
