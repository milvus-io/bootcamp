import React from "react";
import { X } from "react-feather";
import styled from "styled-components";
import { Button } from "reakit";

type uploaderProps = {
  isActive: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title?: string;
};

const UploadDialog: React.FC<uploaderProps> = ({
  isActive,
  onClose,
  onConfirm,
  title,
  children,
}) => {
  const handleCloseDialog = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };
  return (
    <>
      {isActive && (
        <UploaderWrapper onClick={handleCloseDialog}>
          <UploaderContent>
            {title && <Title>{title}</Title>}
            <CloseIcon onClick={() => onClose()} />
            <ContentWrapper>{children}</ContentWrapper>
            <ButtonWrapper>
              <CancelButton onClick={() => onClose()}>Cancel</CancelButton>
              <ConfirmButton onClick={() => onConfirm()}>Confirm</ConfirmButton>
            </ButtonWrapper>
          </UploaderContent>
        </UploaderWrapper>
      )}
    </>
  );
};

export default UploadDialog;

const UploaderWrapper = styled.div`
  position: fixed;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
`;

const UploaderContent = styled.div`
  width: 50%;
  height: 50%;
  background: #fff;
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
`;

const CloseIcon = styled(X)`
  display: inline;
  height: 16px;
  width: 16px;
  margin-right: 8px;
  position: absolute;
  right: 16px;
  top: 16px;
  cursor: pointer;
`;

const ButtonWrapper = styled.div`
  display: flex;
  position: absolute;
  padding: 32px;
  right: 0;
  bottom: 0;
  width: 100%;
  justify-content: flex-end;
`;

const ConfirmButton = styled(Button)`
  display: flex;
  background: #4eb8f0;
  border: none;
  padding: 12px 16px;
  cursor: pointer;
  color: ${({ theme }) => theme.white};
  border-radius: 4px;
  outline: none;
  transition: background 0.1s;
  margin-left: 16px;
  margin-right: 16px;

  &:hover {
    background: ${({ theme }) => theme.secondary};
  }
`;
const CancelButton = styled(Button)`
  background: #4eb8f0;
  border: none;
  padding: 12px 16px;
  cursor: pointer;
  color: ${({ theme }) => theme.white};
  border-radius: 4px;
  outline: none;
  transition: background 0.1s;
  margin-left: 16px;
  margin-right: 16px;

  &:hover {
    background: ${({ theme }) => theme.secondary};
  }
`;

const Title = styled.p`
  position: absolute;
  top: 16px;
  left: 16px;
`;

const ContentWrapper = styled.div`
  padding: 32px;
  width: 100%;
  height: calc(100% - 107px);
  display: flex;
  justify-content: center;
  align-items: center;
`;
