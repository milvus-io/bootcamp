// Definitions by: Wu Haotian <https://github.com/whtsky>
export interface IUserOptions {
    allAtOnce?: boolean;
    delay?: number;
}

type TargetElement = Element | Window;

declare const userEvent: {
    click: (element: TargetElement) => void;
    dblClick: (element: TargetElement) => void;
    selectOptions: (element: TargetElement, values: string | string[]) => void;
    type: (
        element: TargetElement,
        text: string,
        userOpts?: IUserOptions
    ) => Promise<void>;
};

export default userEvent;
