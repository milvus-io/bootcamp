export const getMoviesFromRes = (data) => {
  const movies = Object.entries(data).reduce((acc, cur) => {
    const [key, value] = cur;
    const [name, year, imgUrl] = value;
    acc = [
      ...acc,
      {
        name,
        year,
        imgUrl,
        id: Number(key),
      },
    ];
    return acc;
  }, []);

  return movies;
};
