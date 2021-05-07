import { Card, CardContent, IconButton } from '@material-ui/core';
import React, { useEffect, useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import ArrowBackIosIcon from '@material-ui/icons/ArrowBackIos';
import './index.css';
import { useHistory, useLocation } from 'react-router';
import { sendRequest } from '../../shared/http.util';
import { getMoviesFromRes } from '../../shared/format.util';
import Loading from '../../components/Loading';

const useStyles = makeStyles({
  card: {
    position: 'relative',

    fontSize: '1.5rem',
    fontWeight: '500',
    color: '#fff',

    margin: '1rem',
    minHeight: '20vh',
  },
  back: {
    position: 'absolute',
    top: '0.5rem',
    left: '0.5rem',

    color: '#fff',
  },
});

const RecommendPage = () => {
  const classes = useStyles();
  const location = useLocation();
  const history = useHistory();
  const [recommendMovies, setRecommendMovies] = useState([]);
  const [isLoaded, setIsLoaded] = useState(false);

  const fetchRecommendMovies = async (ids) => {
    try {
      const response = await sendRequest('POST', 'getSimilarUser', null, ids);
      const movies = getMoviesFromRes(response.data);
      setRecommendMovies(movies);
      setIsLoaded(true);
    } catch (err) {
      setRecommendMovies([]);
      setIsLoaded(true);
      throw err;
    }
  };

  const fetchRandomMovies = async () => {
    try {
      const response = await sendRequest(
        'POST',
        'getRandom',
        { num: 100 },
        null
      );
      const movies = getMoviesFromRes(response.data);
      setRecommendMovies(movies);
      setIsLoaded(true);
    } catch (err) {
      setRecommendMovies([]);
      setIsLoaded(true);
      throw err;
    }
  };

  /* fetch recommend movie list */
  useEffect(() => {
    const likedMovieIds = location.state.ids;

    if (likedMovieIds.length > 0) {
      fetchRecommendMovies(likedMovieIds);
    } else {
      fetchRandomMovies();
    }
  }, [location.state.ids]);

  const onCardClick = (movie) => {
    history.push({
      pathname: '/detail',
      state: {
        id: movie.id,
      },
    });
  };

  const onBackClick = () => {
    history.goBack();
  };

  return (
    <section className="recommend-wrapper">
      <IconButton className={classes.back} onClick={onBackClick}>
        <ArrowBackIosIcon />
      </IconButton>

      <h3>Movie Recommendations</h3>

      {!isLoaded && <Loading />}

      <section className="cards-wrapper">
        {recommendMovies.map((movie) => {
          return (
            <Card
              key={movie.id}
              className={classes.card}
              onClick={() => onCardClick(movie)}
              style={{
                backgroundImage: `url(${movie.imgUrl})`,
                backgroundSize: 'cover',
              }}
            >
              <div className="card-mask">
                <CardContent>
                  {movie.name} ({movie.year})
                </CardContent>
              </div>
            </Card>
          );
        })}
      </section>
    </section>
  );
};

export default RecommendPage;
