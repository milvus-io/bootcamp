import { Chip, IconButton, makeStyles } from '@material-ui/core';
import { Rating } from '@material-ui/lab';
import ArrowBackIosIcon from '@material-ui/icons/ArrowBackIos';
import React, { useEffect, useState } from 'react';
import { useHistory, useLocation } from 'react-router';
import { sendRequest } from '../../shared/http.util';
import './index.css';
import Loading from '../../components/Loading';

const useStyles = makeStyles({
  back: {
    position: 'absolute',
    top: '0.5rem',
    left: '0.5rem',
    color: '#fff',
    zIndex: '99',
  },

  chip: {
    marginRight: '0.5rem',
    marginBottom: '0.5rem',
  },

  rate: {
    marginRight: '0.5rem',
    fontSize: '0.8rem',
  },
});

const getDetailFromRes = (res) => {
  const [data, imgUrl] = res;
  const detail = {
    ...data,
    genres: data.Genre.split(','),
    imgUrl,
  };

  return detail;
};

const DetailPage = () => {
  const classes = useStyles();
  const location = useLocation();
  const history = useHistory();

  const [detail, setDetail] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);

  const fetchMovieDetail = async (id) => {
    try {
      const response = await sendRequest('GET', 'getInfo', { ids: id }, null);
      const detail = getDetailFromRes(response.data);
      setDetail(detail);
      setIsLoaded(true);
    } catch (err) {
      setDetail(null);
      setIsLoaded(true);
      throw err;
    }
  };

  useEffect(() => {
    const id = location.state.id;
    fetchMovieDetail(id);
  }, [location.state.id]);

  const onBackClick = () => {
    history.goBack();
  };

  return (
    <section className="detail-wrapper">
      <IconButton className={classes.back} onClick={onBackClick}>
        <ArrowBackIosIcon />
      </IconButton>

      {!isLoaded && <Loading />}

      {detail && (
        <section className="detail-info">
          <div className="detail-hero">
            <img
              className="detail-img"
              src={detail.imgUrl}
              alt="movie poster"
            />
            <div className="detail-img-mask"></div>
            <h4 className="detail-title">
              {detail.Title} ({detail.Year})
            </h4>
          </div>

          <div className="detail-content">
            <div className="detail-genre">
              {detail.genres.map((genre) => (
                <Chip className={classes.chip} key={genre} label={genre} />
              ))}
            </div>

            <div className="detail-item-wrapper">
              <span className="detail-label">IMDB:</span>
              <div className="detail-rating-wrapper">
                <div className="detail-rating-point">{detail.imdbRating}</div>
                <div>
                  <Rating
                    className={classes.rate}
                    value={detail.imdbRating / 2}
                    readOnly
                  />
                  <div className="detail-rating-votes">
                    {detail.imdbVotes} votes
                  </div>
                </div>
              </div>
            </div>
            <div className="detail-item-wrapper">
              <span className="detail-label">Runtime:</span> {detail.Runtime}
            </div>
            <div className="detail-item-wrapper">
              <span className="detail-label">Release Date:</span>{' '}
              {detail.Released}
            </div>

            <p className="detail-plot">{detail.Plot}</p>
          </div>
        </section>
      )}
    </section>
  );
};

export default DetailPage;
