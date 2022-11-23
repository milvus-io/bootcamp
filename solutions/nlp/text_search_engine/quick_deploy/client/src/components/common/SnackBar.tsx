import React, { useEffect } from "react";
import styled from "styled-components";

type snackBarProps = {
  isActive: boolean;
  onClose: () => void;
  duration?: number;
  type?: string;
  content: string;
};

const SnackBar: React.FC<snackBarProps> = ({
  isActive,
  onClose,
  duration = 5,
  type,
  content,
}) => {
  useEffect(() => {
    let timer: any = null;

    if (isActive) {
      timer = setTimeout(() => {
        onClose();
      }, duration * 1000);
    }

    return () => {
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [isActive, onClose, duration]);

  return (
    <>
      {isActive && (
        <SnackBarWrapper
          color={`${type === "info" ? "#d9f7be" : "#ffccc7"}`}
          // backgroundColor={`${type === "info" ? "#d9f7be" : "#ffccc7"}`}
          onClick={() => {
            onClose();
          }}
        >
          {content}
        </SnackBarWrapper>
      )}
    </>
  );
};

export default SnackBar;

const SnackBarWrapper = styled.div`
  position: fixed;
  right: 0;
  top: 100px;
  color: #424242;
  background-color: ${(props) => props.color};
  padding: 16px;
`;
