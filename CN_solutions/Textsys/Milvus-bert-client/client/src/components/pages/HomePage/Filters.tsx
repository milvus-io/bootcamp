import React from 'react';
import styled from 'styled-components';
import RangeSlider from '../../common/RangeSlider';
import { Heading2, Body } from '../../../shared/Styles';
import SelectionFilter from '../../common/SelectionFilter';
import { SearchFilters, SelectedSearchFilters } from '../../../shared/Models';

interface FiltersProps {
  filters: SearchFilters;
  selectedFilters: SelectedSearchFilters;
  setSelectedFilters: (value: SelectedSearchFilters) => void;
}

const updateSelectionFilter = (selectedFilter: Set<string>, value: string): Set<string> => {
  let newFilters = selectedFilter;
  if (newFilters.has(value)) {
    newFilters.delete(value);
  } else {
    newFilters.add(value);
  }
  return newFilters;
};

const Filters: React.FC<FiltersProps> = ({ filters, selectedFilters, setSelectedFilters }) => {
  return (
    <FiltersWrapper>
      <FilterTitle>Filter your search</FilterTitle>
      {filters.yearMinMax[0] < filters.yearMinMax[1] && (
        <FilterComponent>
          <FilterSubtitle>Publication Time</FilterSubtitle>
          <RangeSlider
            min={filters.yearMinMax[0]}
            max={filters.yearMinMax[1]}
            values={selectedFilters.yearRange}
            setValues={(values) =>
              setSelectedFilters({
                ...selectedFilters,
                yearRange: values,
              })
            }
          />
        </FilterComponent>
      )}
      {filters.sources.length > 0 && (
        <FilterComponent>
          <FilterSubtitle>Source</FilterSubtitle>
          <SelectionFilter
            options={filters.sources}
            selectedOptions={selectedFilters.sources}
            maxDisplayed={10}
            setSelectedOptions={(source) => {
              setSelectedFilters({
                ...selectedFilters,
                sources: updateSelectionFilter(selectedFilters.sources, source),
              });
            }}
          />
        </FilterComponent>
      )}
      {filters.authors.length > 0 && (
        <FilterComponent>
          <FilterSubtitle>Author</FilterSubtitle>
          <SelectionFilter
            options={filters.authors}
            selectedOptions={selectedFilters.authors}
            setSelectedOptions={(author) => {
              setSelectedFilters({
                ...selectedFilters,
                authors: updateSelectionFilter(selectedFilters.authors, author),
              });
            }}
          />
        </FilterComponent>
      )}
      {filters.journals.length > 0 && (
        <FilterComponent>
          <FilterSubtitle>Journal</FilterSubtitle>
          <SelectionFilter
            options={filters.journals}
            selectedOptions={selectedFilters.journals}
            setSelectedOptions={(journal) => {
              setSelectedFilters({
                ...selectedFilters,
                journals: updateSelectionFilter(selectedFilters.journals, journal),
              });
            }}
          />
        </FilterComponent>
      )}
    </FiltersWrapper>
  );
};

export default Filters;

const FiltersWrapper = styled.div`
  display: flex;
  flex-direction: column;
  width: 200px;
  min-width: 200px;
  margin-right: 48px;
  padding-top: 24px;

  @media only screen and (max-width: ${({ theme }) => theme.breakpoints.singleColumn}px) {
    width: 100%;
    padding: 8px 0;
    border-bottom: 1px solid ${({ theme }) => theme.lightGrey};
  }
`;

const FilterTitle = styled.div`
  ${Heading2}
  margin-bottom: 16px;
`;

const FilterSubtitle = styled.div`
  ${Body}
  color: ${({ theme }) => theme.darkGrey};
  font-weight: 600;
`;

const FilterComponent = styled.div`
  margin: 16px auto;
  width: 100%;

  @media only screen and (max-width: ${({ theme }) => theme.breakpoints.singleColumn}px) {
    margin: 8px auto;
  }
`;
