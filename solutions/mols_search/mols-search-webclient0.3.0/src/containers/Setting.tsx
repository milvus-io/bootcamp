import React, { useState, useEffect, useContext } from "react";
import { queryContext } from "../contexts/QueryContext";
import { makeStyles } from "@material-ui/core/styles";
import TextField from "@material-ui/core/TextField";
import SearchIcon from "@material-ui/icons/Search";
import InputAdornment from "@material-ui/core/InputAdornment";
import Fab from "@material-ui/core/Fab";
import AddIcon from "@material-ui/icons/Add";
import Slider from "@material-ui/core/Slider";
import SeperatLine from "../components/SeperatLine";
import { baseColor } from "../utils/color";
import Logo from "./Logo.svg";
import { delayRunFunc } from "../utils/Helper";

const useStyles = makeStyles({
  setting: {
    display: "flex",
    flexDirection: "column",
    minWidth: "400px",
    padding: "60px 20px",
    borderWidth: "1px",
    backgroundColor: "#1F2023",
    color: "#E4E4E6",
    overflowY: "auto"
  },
  header: {
    marginBottom: "30px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center"
  },
  config: {
    fontSize: "24px",
    color: "#FAFAFA"
  },
  clear: {
    color: baseColor,
    fontSize: "18px",
    cursor: "pointer"
  },
  imageSet: {},
  counts: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "30px",
    color: "#FAFAFA"
  },
  currTotal: {
    fontSize: "12px"
  },
  setPath: {
    display: "flex",
    justifyContent: "start",
    alignItems: "center",
    marginBottom: "60px"
  },
  MolecularInput: {
    margin: "0 0 40px 0 !important",
    color: "blue !important"
  },
  customInput: {
    margin: "0 20px 0 0 !important",
    color: "blue !important"
  },
  customFab: {
    color: "#fff",
    backgroundColor: baseColor,
    width: "36px",
    height: "36px",
    "&:hover": {
      backgroundColor: baseColor
    }
  },
  customDeleteFab: {
    position: "absolute",
    top: "5px",
    right: "5px",
    color: "#fff",
    backgroundColor: "#666769",
    width: "24px",
    height: "24px",
    minHeight: "0px",
    "&:hover": {
      backgroundColor: "#666769"
    }
  },
  customDelete: {
    color: "#A7A7AF",
    width: "18px",
    height: "18px"
  },
  customIcon: {
    color: "#fff",
    backgroundColor: baseColor,
    width: "20px",
    height: "20px"
  },
  customSlider: {
    color: baseColor,
    marginBottom: "30px"
  },
  thumb: {
    width: "16px",
    height: "16px"
  },
  track: {
    height: "4px",
    borderRadius: "10px"
  },
  upload: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center"
  },
  benchImage: {
    width: "400px",
    height: "250px",
    position: "relative"
  },
  dropzoneContainer: {
    backgroundColor: "transparent",
    width: "250px",
    height: "250px",
    borderRadius: "10px",
    border: "solid .5px #C8C8C8",
    display: "flex",
    justifyContent: "center",
    alignItems: "center"
  },
  dropzoneText: {
    fontSize: "14px",
    color: "#B3B4B5",
    marginBottom: "30px"
  },
  notchedOutline: {
    borderWidth: ".5px",
    borderColor: "#838385 !important"
  },
  formLabel: {
    color: "#fff"
  },
  controlLabel: {
    color: "#838385"
  }
});
// /data/workspace/apptec/demo/test_100.smi
// COc1ccc(cc1)SCCC(=O)NCCNS(=O)(=O)c1cccc(c1)Cl
const Setting = (props: any) => {
  const { showNote, load, process, count, search, clearAll } = useContext(
    queryContext
  );
  const { setResults, loading, setLoading } = props;
  const classes = useStyles({});
  const [inputs, setInputs]: any = useState("");
  const [topK, setTopK]: any = useState(5);
  const [totalNum, setTotalNum]: any = useState(0);
  const [[current, total], setProcessedNum]: any = useState([0, 0]);
  const [Molecular, setMolecular]: any = useState();

  const setText = loading
    ? "Loading..."
    : totalNum
    ? `${totalNum} Molecular Formula in this set`
    : "No Molecular Formula in this set";

  const _search = ({ topK, Molecular }: any) => {
    search({ Num: topK, Molecular }).then((res: any) => {
      const { status, data } = res || {};
      if (status === 200) {
        console.log(data, typeof data);
        typeof data === "string" ? showNote(data) : setResults(data);
      }
    });
  };
  const _changeFormula = (e: any) => {
    const val = e.target.value;
    setMolecular(val);
  };
  const onInputChange = (e: any) => {
    const val = e.target.value;
    setInputs(val);
  };

  const onTopKChange = (e: any, val: any) => {
    setTopK(val);
    if (val && Molecular) {
      delayRunFunc({ topK: val, Molecular }, _search, 300);
    }
  };
  const _keepProcess = () => {
    process().then((res: any) => {
      const { data, status } = res;
      if (status === 200) {
        const [_current, _total] = data
          .split(",")
          .map((item: any) => Number.parseInt(item.split(":")[1]));
        setProcessedNum([_current, _total]);
        if (_current !== _total) {
          setTimeout(() => _keepProcess(), 1000);
        } else {
          setTimeout(() => {
            count().then((res: any) => {
              const { data, status } = res;
              if (status === 200) {
                setTotalNum(data);
                setLoading(false);
              }
            });
          }, 1000);
        }
      }
    });
  };
  const _uploadSetPath = () => {
    load({ File: inputs }).then((res: any) => {
      if (res.status === 200) {
        setLoading(true);
        setTimeout(() => {
          setInputs("");
          _keepProcess();
        }, 1000);
      }
    });
  };

  const clear = () => {
    clearAll().then((res: any) => {
      if (res.status === 200) {
        setProcessedNum([0, 0]);
        setTotalNum(0);
        setMolecular();
      }
    });
  };

  useEffect(() => {
    count().then((res: any) => {
      const { data, status } = res || {};
      if (status === 200) {
        setTotalNum(data);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={classes.setting}>
      <div className={classes.header}>
        <img src={Logo} width="150px" alt="logo" />
        <h3 style={{ marginBottom: "10px" }}>Molecular Formula Search</h3>
        <p>For Research Properly Only</p>
      </div>
      <TextField
        classes={{ root: classes.MolecularInput }}
        label=""
        variant="outlined"
        value={Molecular || ""}
        onChange={_changeFormula}
        InputLabelProps={{
          shrink: true,
          classes: {
            root: classes.controlLabel,
            focused: classes.controlLabel
          }
        }}
        margin="normal"
        InputProps={{
          style: {
            textAlign: "left",
            width: "400px",
            height: "40px"
          },
          classes: {
            notchedOutline: classes.notchedOutline,
            root: classes.formLabel
          },
          placeholder: "please input your chemistry",
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
          onKeyPress: (e: any) => {
            if (e.key === "Enter") {
              _search({ topK, Molecular });
            }
          }
        }}
      />
      <SeperatLine
        title={`CONFIG`}
        end="CLEAR ALL"
        onEndClick={clear}
        style={{ marginBottom: "30px" }}
      />
      <div className={classes.imageSet}>
        <div className={classes.counts}>
          <p style={{ color: loading ? baseColor : "#fff" }}>{setText}</p>
          <h3 className={classes.currTotal}>{`${current}/${total}`}</h3>
        </div>
        <div className={classes.setPath}>
          <TextField
            classes={{ root: classes.customInput }}
            label=""
            variant="outlined"
            value={inputs}
            onChange={onInputChange}
            InputLabelProps={{
              shrink: true,
              classes: {
                root: classes.controlLabel,
                focused: classes.controlLabel
              }
            }}
            margin="normal"
            InputProps={{
              style: {
                textAlign: "left",
                width: "340px",
                height: "40px"
              },
              classes: {
                notchedOutline: classes.notchedOutline,
                root: classes.formLabel
              },
              placeholder: "path/to/your/data"
            }}
          />
          <Fab
            classes={{
              root: classes.customFab,
              focusVisible: classes.customFab
            }}
          >
            <AddIcon
              onClick={_uploadSetPath}
              classes={{ root: classes.customIcon }}
            />
          </Fab>
        </div>
        <SeperatLine
          title={`TopK = ${topK}`}
          style={{ marginBottom: "20px" }}
        />
        <div className={classes.counts}>
          <p>{`show top ${topK} results`}</p>
        </div>
        <Slider
          min={1}
          max={100}
          value={topK}
          onChange={onTopKChange}
          classes={{
            root: classes.customSlider,
            track: classes.track,
            rail: classes.track,
            thumb: classes.thumb
          }}
        />
      </div>
    </div>
  );
};

export default Setting;
