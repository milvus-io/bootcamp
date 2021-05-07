import React, { useState, useEffect } from 'react';
import './index.css';
import { sendRequest } from '../../shared/http.util';

import SentimentSatisfiedOutlinedIcon from '@material-ui/icons/SentimentSatisfiedOutlined';
import SentimentDissatisfiedOutlinedIcon from '@material-ui/icons/SentimentDissatisfiedOutlined';

import { Button, makeStyles } from '@material-ui/core';
import { useHistory } from 'react-router';
import { getMoviesFromRes } from '../../shared/format.util';
import Loading from '../../components/Loading';

const useStyles = makeStyles({
  button: {
    color: '#fff',
    marginBottom: '8px',
    width: '80%',
  },

  icon: {
    color: '#fff',
    fontSize: '4rem',
  },
});

const getBackgroundImgStyle = (url) => ({
  backgroundImage: `url(${url})`,
  backgroundSize: 'cover',

  position: 'absolute',
  top: 0,
  bottom: 0,
  left: 0,
  right: 0,

  filter: `grayscale(0.8) blur(10px)`,
  zIndex: 5,
});

const getStyleByIndex = (movies, index) => {
  const { imgUrl } = movies[index];
  const backgroundStyle = getBackgroundImgStyle(imgUrl);
  return backgroundStyle;
};

const HomePage = () => {
  const classes = useStyles();
  const history = useHistory();

  const [randomMovies, setRandomMovies] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [likedMovieIds, setLikedMovieIds] = useState([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const [wrapperStyle, setWrapperStyle] = useState({});

  const fetchRandomMovies = async () => {
    try {
      const response = await sendRequest('POST', 'getRandom', null, null);
      const movies = getMoviesFromRes(response.data);
      setRandomMovies(movies);
      const style = getStyleByIndex(movies, 0);
      setWrapperStyle(style);
      sessionStorage.setItem('randomMovies', JSON.stringify(movies));
      setIsLoaded(true);
    } catch (err) {
      setIsLoaded(true);
      setRandomMovies([]);
      throw err;
    }
  };

  /* fetch random movie list*/
  useEffect(() => {
    const movies = sessionStorage.getItem('randomMovies');
    const activeIndex = sessionStorage.getItem('activeIndex');
    if (movies) {
      const randomMovies = JSON.parse(movies);
      setRandomMovies(randomMovies);
      const index = Number(activeIndex);
      const style = getStyleByIndex(randomMovies, index);
      setWrapperStyle(style);
      setActiveIndex(index);
      setIsLoaded(true);
    } else {
      fetchRandomMovies();
    }
  }, []);

  const onLikeButtonClick = (event) => {
    event.stopPropagation();

    const { id } = randomMovies[activeIndex];

    if (!likedMovieIds.includes(id)) {
      const movieIds = [...likedMovieIds, id];
      setLikedMovieIds(movieIds);
    }

    goToNextMovie();
  };

  const onDislikeButtonClick = (event) => {
    event.stopPropagation();
    goToNextMovie();
  };

  const onViewDetailClick = (movieId, index) => {
    history.push({
      pathname: '/detail',
      state: {
        id: movieId,
      },
    });

    sessionStorage.setItem('activeIndex', index);
  };

  const goToNextMovie = () => {
    /*
      the number of movies to test your taste is 16 
      so we use index 15 to decide whether jumping to recommend page
    */

    if (activeIndex < 15) {
      const nextIndex = activeIndex + 1;
      setActiveIndex(nextIndex);

      const style = getStyleByIndex(randomMovies, nextIndex);
      setWrapperStyle(style);
    } else {
      submitLikedMovies(likedMovieIds);
    }
  };

  const submitLikedMovies = (ids) => {
    history.push({
      pathname: '/recommend',
      state: { ids },
    });
  };

  return (
    <section className="home-wrapper">
      <div style={wrapperStyle}></div>
      <section className="home-container">
        {!isLoaded && <Loading />}

        {randomMovies.length > 0 &&
          randomMovies.map((movie, index) => {
            return (
              <div
                className="movie-wrapper"
                key={movie.name}
                onClick={() => {
                  onViewDetailClick(movie.id, index);
                }}
                style={{
                  display: index === activeIndex ? 'block' : 'none',
                }}
              >
                <div className="movie-title">
                  {movie.name} ({movie.year})
                </div>

                <img
                  className="movie-poster"
                  src={movie.imgUrl}
                  alt="movie poster"
                />
              </div>
            );
          })}

        <div className="movie-button">
          <Button
            variant="contained"
            color="primary"
            className={classes.button}
            startIcon={
              <SentimentSatisfiedOutlinedIcon className={classes.icon} />
            }
            onClick={onLikeButtonClick}
          >
            LIKE
          </Button>

          <Button
            variant="contained"
            color="secondary"
            className={classes.button}
            startIcon={
              <SentimentDissatisfiedOutlinedIcon className={classes.icon} />
            }
            onClick={onDislikeButtonClick}
          >
            DISLIKE
          </Button>
        </div>
      </section>
    </section>
  );
};

export default HomePage;
