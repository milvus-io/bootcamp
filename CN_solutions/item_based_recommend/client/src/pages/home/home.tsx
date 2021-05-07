import React, { useEffect, useState } from 'react';
import { useHistory } from 'react-router';
import styled from 'styled-components';
import Header from '../../components/header';
import Loading from '../../components/loading';
import { IParam } from '../../shared/common';
import { sendRequest } from '../../shared/http.util';

const HomeWrapper = styled.section`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

  margin: 80px 0 80px 0;
`;

const ContentWrapper = styled.main`
  display: grid;
  grid-template-columns: repeat(auto-fill, 150px);
  grid-gap: ${({ theme }) => theme.spacing.lg};
  justify-content: center;

  margin-top: 80px;
  max-width: 648px;
  width: 100%;
`;

const CategoryCard = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;

  height: 150px;
  border-radius: 8px;

  background-color: ${({ theme }) => theme.color.primary};
  color: #fff;

  &:hover {
    cursor: pointer;
  }
`;

const HomePage = () => {
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [categories, setCategories] = useState<string[]>([]);

  const history = useHistory();

  const fetchCategories = async () => {
    try {
      const param: IParam = {
        path: 'categories',
        method: 'GET',
      };
      const response = await sendRequest(param);
      setIsLoaded(true);
      const categories = response.data.map((item: any[]) => {
        const [, name] = item;
        return name;
      });
      setCategories(categories);
    } catch (err) {
      setIsLoaded(true);
      throw err;
    }
  };

  useEffect(() => {
    fetchCategories();
  }, []);

  const onCategoryClick = (category: string) => {
    history.push({
      pathname: '/recommend',
      state: {
        category,
        categories,
      },
    });
  };

  return (
    <HomeWrapper>
      <Header />

      {!isLoaded && <Loading />}

      <ContentWrapper>
        {categories.length > 0 &&
          categories.map((category) => (
            <CategoryCard
              key={category}
              onClick={() => onCategoryClick(category)}
            >
              {category}
            </CategoryCard>
          ))}
      </ContentWrapper>
    </HomeWrapper>
  );
};

export default HomePage;
