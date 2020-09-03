import React from 'react';
import { useLocation } from 'react-router';
import './index.css';

const Footer = () => {
  const location = useLocation();

  return (
    <footer
      className="wrapper"
      style={{
        backgroundColor: location.pathname === '/' ? 'transparent' : '#565f77',
      }}
    >
      <div className="title">Movie Taste</div>
      <div className="subtitle">Powered by Milvus</div>
    </footer>
  );
};

export default Footer;
