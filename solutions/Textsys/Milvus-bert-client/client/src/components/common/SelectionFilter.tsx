import React, { useState, ChangeEvent } from 'react';
import styled, { useTheme } from 'styled-components';
import { Body, LinkStyle, BodySmall } from '../../shared/Styles';
import { Check } from 'react-feather';

interface SelectionFilterProps {
  options: string[];
  selectedOptions: Set<string>;
  setSelectedOptions: (value: string) => void;
  maxDisplayed?: number;
}

const SelectionFilter: React.FC<SelectionFilterProps> = ({
  options = [],
  selectedOptions = new Set([]),
  setSelectedOptions = () => {},
  maxDisplayed = 5,
}) => {
  const theme = useTheme();

  const [numberDisplayed, setNumberDisplayed] = useState<number>(maxDisplayed);
  const [optionQuery, setOptionQuery] = useState<string>('');

  const notSelected = options
    .filter((option) => !selectedOptions.has(option) && option !== '')
    .sort();
  const sortedOptions: string[] = [...Array.from(selectedOptions).sort(), ...notSelected];
  const filteredOptions = sortedOptions.filter(
    (option) => optionQuery === '' || option.toLowerCase().includes(optionQuery.toLowerCase()),
  );

  return (
    <FilterWrapper>
      {sortedOptions.length > maxDisplayed && (
        <QueryInput
          placeholder="Find a filter..."
          value={optionQuery}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setOptionQuery(e.target.value)}
        />
      )}
      {filteredOptions.slice(0, numberDisplayed).map((option, idx) => {
        return (
          <OptionWrapper
            key={idx}
            onClick={() => setSelectedOptions(option)}
            onMouseDown={(e: any) => e.preventDefault()}
          >
            <Checkbox className="checkbox" checked={selectedOptions.has(option)}>
              {selectedOptions.has(option) && (
                <Check strokeWidth={3} color={theme.white} size={14} />
              )}
            </Checkbox>
            <OptionText>{option}</OptionText>
          </OptionWrapper>
        );
      })}
      {filteredOptions.length > numberDisplayed && (
        <ShowMoreText
          onClick={() => setNumberDisplayed(numberDisplayed + 10)}
          onMouseDown={(e: any) => e.preventDefault()}
        >
          More filters...
        </ShowMoreText>
      )}
    </FilterWrapper>
  );
};

export default SelectionFilter;

const FilterWrapper = styled.div`
  display: flex;
  flex-direction: column;
  margin-top: 16px;
`;

const QueryInput = styled.input`
  ${Body}
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid ${({ theme }) => theme.grey};
  color: ${({ theme }) => theme.slate};
  margin-bottom: 16px;
  outline: none;

  &:focus {
    border: 1px solid ${({ theme }) => theme.primary};
  }
`;

const OptionWrapper = styled.button`
  display: flex;
  text-align: left;
  margin-bottom: 8px;
  cursor: pointer;
  border: none;
  padding: 0;
  background: ${({ theme }) => theme.white};

  &:hover {
    & > .checkbox {
      filter: brightness(95%);
    }
  }
`;

const Checkbox = styled.div<{ checked: boolean }>`
  cursor: pointer;
  border: 1px solid ${({ theme, checked }) => (checked ? theme.primary : theme.grey)};
  margin-right: 8px;
  background: ${({ theme, checked }) => (checked ? theme.primary : theme.white)};
  cursor: pointer;
  width: 16px;
  min-width: 16px;
  height: 16px;
  border-radius: 4px;
`;

const OptionText = styled.div`
  ${BodySmall}
  color: ${({ theme }) => theme.primary};
  cursor: pointer;

  &:hover {
    color: ${({ theme }) => theme.secondary};
  }
`;

const ShowMoreText = styled.button`
  ${LinkStyle}
  ${BodySmall}
  margin-top: 8px;
  border: none;
  padding: 0;
  background: transparent;
  text-align: left;
`;
