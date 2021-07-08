import React from 'react';
import { Range, getTrackBackground } from 'react-range';
import styled, { useTheme } from 'styled-components';
import { BodySmall } from '../../shared/Styles';

interface RangeSliderProps {
  min: number;
  max: number;
  step?: number;
  values: number[];
  setValues: (values: number[]) => void;
  monthLabels?: boolean;
}

const RangeSlider: React.FC<RangeSliderProps> = ({
  min = 0,
  max = 0,
  step = 1,
  values,
  setValues,
}) => {
  const theme = useTheme();
  const minString = String(values[0]);
  const maxString = String(values[1]);

  return (
    <RangeWrapper>
      <Range
        step={step}
        min={min}
        max={max}
        values={values}
        onChange={setValues}
        renderTrack={({ props, children }) => (
          <Track
            {...props}
            style={{
              background: getTrackBackground({
                values,
                colors: [theme.grey, theme.primary, theme.grey],
                min: min,
                max: max,
              }),
            }}
          >
            {children}
          </Track>
        )}
        renderThumb={({ props }) => <Thumb {...props} />}
      />
      <Values>
        <ValueText>
          {minString.length > 4 ? `${minString.substr(4, 6)}/${minString.substr(0, 4)}` : minString}
        </ValueText>
        <ValueText>
          {maxString.length > 4 ? `${maxString.substr(4, 6)}/${maxString.substr(0, 4)}` : maxString}
        </ValueText>
      </Values>
    </RangeWrapper>
  );
};

export default RangeSlider;

const THUMB_WIDTH = 28;

const RangeWrapper = styled.div`
  margin-top: ${THUMB_WIDTH / 2 + 8}px;
`;

const Track = styled.div`
  height: 8px;
  border-radius: 4px;
  margin: 0 ${THUMB_WIDTH / 2}px;
`;

const Thumb = styled.div`
  width: ${THUMB_WIDTH}px;
  height: ${THUMB_WIDTH}px;
  background: ${({ theme }) => theme.primary};
  border-radius: 50%;
  border: 2px solid ${({ theme }) => theme.lightGrey};
  outline: none;
  position: relative;

  &:hover,
  &:focus {
    background: ${({ theme }) => theme.secondary};
  }
`;

const Values = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: ${THUMB_WIDTH / 2}px;
`;

const ValueText = styled.div`
  ${BodySmall}
  color: ${({ theme }) => theme.secondary};
`;
