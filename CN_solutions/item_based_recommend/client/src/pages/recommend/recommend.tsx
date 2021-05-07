import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router';
import styled from 'styled-components';
import Drawer from '@material-ui/core/Drawer';
import Loading from '../../components/loading';
import { IParam } from '../../shared/common';
import { sendRequest } from '../../shared/http.util';
import { ICategory, IRouteParam, IText } from './types';
import Header from '../../components/header';
import DoubleArrowIcon from '@material-ui/icons/DoubleArrow';

const RecommendWrapper = styled.section`
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
`;

const ContentWrapper = styled.div`
  display: flex;
  max-width: 1170px;

  height: 90vh;
`;

const CategoryMenu = styled.div`
  flex-shrink: 0;
  margin-right: 32px;
`;

const CategoryItem = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;

  height: 48px;
  padding: 0 8px;
  border-radius: 8px;

  margin-bottom: ${({ theme }) => theme.spacing.sm};
  cursor: pointer;

  &:hover {
    cursor: pointer;
    color: #fff;
    background-color: ${({ theme }) => theme.color.primary};
  }
`;

const CategoryActiveItem = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;

  height: 48px;
  padding: 0 8px;
  border-radius: 8px;

  margin-bottom: ${({ theme }) => theme.spacing.sm};

  color: #fff;
  background-color: #4eb8f0;

  cursor: pointer;
`;

const PaperWrapper = styled.div`
  padding: 48px 16px;
`;

const PaperItem = styled.div`
  max-width: 560px;
  margin-bottom: 48px;
  padding: 16px;
`;

const TextWrapper = styled.div`
  flex-grow: 1;
  overflow-y: scroll;
`;

const TextItem = styled.div`
  margin: 16px 0;
`;

const TextTitle = styled.a`
  font-weight: 500;
  font-size: 24px;
  text-decoration: none;
  color: #000;

  &:hover {
    text-decoration: underline;
  }
`;

const TextDesc = styled.div`
  -webkit-line-clamp: 4;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;

  margin-top: 8px;
  color: #777;
`;

const RelatedLink = styled.div`
  display: flex;
  align-items: center;
  margin-top: 8px;
  color: ${({ theme }) => theme.color.primary};
  cursor: pointer;
`;

const getTextsFromData = (data: any[]): IText[] => {
  const texts = data.map((item: any) => {
    // id, title, desc, category, href
    const [, title, desc, , href] = item;
    return {
      title,
      desc,
      href,
    };
  });

  return texts;
};

const RecommendPage = () => {
  const location = useLocation();
  const [categories, setCategories] = useState<ICategory[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const [texts, setTexts] = useState<IText[]>([]);
  const [relatedPapers, setRelatedPapers] = useState<IText[]>([]);
  const [isOpenDrawer, setIsOpenDrawer] = useState(false);

  const fetchTexts = async (categoryName: string) => {
    try {
      const param: IParam = {
        path: `category_texts/${categoryName}`,
        method: 'GET',
      };
      const response = await sendRequest(param);
      setIsLoaded(true);
      const { data } = response;
      const texts = getTextsFromData(data);

      setTexts(texts);
    } catch (err) {
      setIsLoaded(true);
      throw err;
    }
  };

  const fetchRelatedPapers = async (desc: string) => {
    try {
      const param: IParam = {
        path: 'search',
        method: 'POST',
        bodyParam: {
          abstract: desc,
        },
      };

      const response = await sendRequest(param);
      const { data } = response;
      const relatedPaperList = getTextsFromData(data);
      setRelatedPapers(relatedPaperList);
      setIsOpenDrawer(true);
    } catch (err) {
      throw err;
    }
  };

  useEffect(() => {
    const { category, categories } = location.state as IRouteParam;
    fetchTexts(category);
    const categoryList = categories.map((item) => ({
      name: item,
      active: item === category,
    }));

    setCategories(categoryList);
  }, [location]);

  const onCategoryClick = (name: string) => {
    const categoryList = categories.map((c) => ({
      ...c,
      active: c.name === name,
    }));

    fetchTexts(name);
    setCategories(categoryList);
  };

  const onRelatedPapaerClick = (desc: string) => {
    fetchRelatedPapers(desc);
  };

  const toggleDrawer = (isOpen: boolean) => (
    event: React.KeyboardEvent | React.MouseEvent
  ) => {
    if (
      event.type === 'keydown' &&
      ((event as React.KeyboardEvent).key === 'Tab' ||
        (event as React.KeyboardEvent).key === 'Shift')
    ) {
      return;
    }
    setIsOpenDrawer(isOpen);
  };

  const getDrawerContent = (papers: IText[]): any => {
    const content = (
      <PaperWrapper>
        {papers.map((paper) => (
          <PaperItem key={paper.title}>
            <TextTitle href={paper.href} target="_blank">
              {paper.title}
            </TextTitle>
            <TextDesc>{paper.desc}</TextDesc>
          </PaperItem>
        ))}
      </PaperWrapper>
    );
    return content;
  };

  return (
    <RecommendWrapper>
      <Header />
      {!isLoaded && <Loading />}
      <ContentWrapper>
        <CategoryMenu>
          {categories.map((category) =>
            category.active ? (
              <CategoryActiveItem key={category.name}>
                {category.name}
              </CategoryActiveItem>
            ) : (
              <CategoryItem
                key={category.name}
                onClick={() => onCategoryClick(category.name)}
              >
                {category.name}
              </CategoryItem>
            )
          )}
        </CategoryMenu>
        <Drawer
          anchor="right"
          open={isOpenDrawer}
          onClose={toggleDrawer(false)}
        >
          {getDrawerContent(relatedPapers)}
        </Drawer>

        <TextWrapper>
          {texts.length > 0 &&
            texts.map((text) => (
              <TextItem key={text.title}>
                <TextTitle href={text.href} target="_blank">
                  {text.title}
                </TextTitle>
                <TextDesc>{text.desc}</TextDesc>
                <RelatedLink onClick={() => onRelatedPapaerClick(text.desc)}>
                  <DoubleArrowIcon fontSize="small" />
                  Related Paper
                </RelatedLink>
              </TextItem>
            ))}
        </TextWrapper>
      </ContentWrapper>
    </RecommendWrapper>
  );
};

export default RecommendPage;
