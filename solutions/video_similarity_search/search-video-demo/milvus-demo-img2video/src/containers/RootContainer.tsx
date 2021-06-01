import React, { useContext } from "react";
import { makeStyles } from "@material-ui/core/styles";
import { queryContext } from "../contexts/QueryContext";
import useMediaQuery from "@material-ui/core/useMediaQuery";
// import Logo from "../logo-white.svg";
import SearchIcon from "@material-ui/icons/Search"
import SettingsIcon from "@material-ui/icons/Settings"
import Library from "../components/Library"
import Search from '../components/Search'
import { CSSTransition, SwitchTransition } from "react-transition-group"

const RootContainer: React.FC = () => {
  const { page, setPage, navTitle, setNavTitle } = useContext(queryContext)
  const isMobile = !useMediaQuery("(min-width:1000px)");
  const useStyles = makeStyles({
    root: {
      flexGrow: 1,
      height: '100vh',
      overflow: 'hidden',
      position: 'relative'
    },
    nav: {
      position: 'relative',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      height: '50px',
      backgroundColor: "#1F2023",
      color: '#fff',
      paddingBottom: '5px',
      borderBottom: "solid 1px rgba(255, 255, 255, .5)",
    },
    title: {
      letterSpacing: 3,
      fontFamily: 'Roboto-Medium,Roboto',
      fontWeight: 500
    },
    logo: {
      paddingLeft: '30px',
    },
    pageSwitcher: {
      marginRight: '30px',
      color: '#fff',
      backgroundColor: 'grey',
      borderRadius: '17px',
      display: 'flex',
      alignItems: 'center',
      cursor: 'pointer'
    },
    note: {
      position: 'absolute',
      fontSize: '4px',
      fontFamily: `Roboto-Regular,Roboto`,
      fontWeight: 400,
      color: `rgba(250,250,250,1)`,
      letterSpacing: `1px`,
      top: 0,
      left: `80px`,
      transform: 'scale(.7)',
      opacity: .8
    },
    selectedWrapper: {
      borderRadius: '50%',
      color: '#000',
      backgroundColor: '#fff',
      padding: '5px'
    },
    noneSelectedWrapper: {
      backgroundColor: 'transparent',
      color: '#fff',
      padding: '5px'
    },
    content: {
      display: isMobile ? 'block' : "flex",
      flexGrow: 1,
      backgroundColor: "#1F2023",
      height: 'calc(100% - 50px)',
      overflowY: 'auto'
    }
  });
  const classes = useStyles({});
  return (
    <div className={classes.root}>
      <div className={classes.nav}>
        {/* <img className={classes.logo} src={Logo} width="150px" alt="logo" /> */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0 20px'
        }}><h3 style={{ marginRight: '10px' }}>Video Search</h3><p> Powered By Milvus</p></div>
        <h3 className={classes.title}>{navTitle}</h3>
        <div className={classes.pageSwitcher} onClick={() => { setPage(page === 'search' ? 'library' : 'search'); page === 'search' && setNavTitle('VIDEO SEARCH'); }}>
          <div className={page === 'search' ? classes.selectedWrapper : classes.noneSelectedWrapper}><SearchIcon style={{ fontSize: '1rem' }} /></div>
          <div className={page === 'library' ? classes.selectedWrapper : classes.noneSelectedWrapper}><SettingsIcon style={{ fontSize: '1rem' }} /></div>
        </div>
        {/* <div className={classes.note}>POWERED BY</div> */}
      </div>
      <div className={classes.content} id='content'>
        <SwitchTransition mode='out-in'>
          <CSSTransition classNames='fade' timeout={500} key={page === 'search' ? "on" : "off"} >
            {page === 'search' ? <Search /> : <Library />}
          </CSSTransition>
        </SwitchTransition>
      </div>
    </div>
  );
};

export default RootContainer;
