import React, { useState } from "react";
import {
  withStyles,
  Theme,
  createStyles,
  makeStyles,
} from "@material-ui/core/styles";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import AudioPlayer from "./audioPlayer";

const StyledTableCell = withStyles((theme: Theme) =>
  createStyles({
    head: {
      background: "#fafafa",
      color: "#999",
    },
    body: {
      fontSize: 14,
    },
  })
)(TableCell);

const StyledTableRow = withStyles((theme: Theme) =>
  createStyles({
    root: {
      "&:nth-of-type(odd)": {
        backgroundColor: theme.palette.action.hover,
      },
    },
  })
)(TableRow);

const useStyles = makeStyles({
  table: {
    minWidth: 700,
  },
  noData: {
    padding: "16px 0",
  },
});

type PropsType = {
  rows: {
    name: string;
    distance: number;
    duration: number;
    audioSrc: string;
  }[];
};

const Search: React.FC<PropsType> = ({ rows }) => {
  const classes = useStyles();

  return (
    <TableContainer component={Paper}>
      <Table className={classes.table} aria-label="customized table">
        <TableHead>
          <TableRow>
            <StyledTableCell>#</StyledTableCell>
            <StyledTableCell align="right">{""}</StyledTableCell>
            <StyledTableCell align="right">Name</StyledTableCell>
            <StyledTableCell align="right">Duration</StyledTableCell>
            <StyledTableCell align="right">Distance</StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.length ? (
            rows.map((row, index) => (
              // <StyledTableRow key={row.name}>
              //   <StyledTableCell>{index + 1}</StyledTableCell>
              //   <StyledTableCell>
              //     <PlayArrowIcon />
              //   </StyledTableCell>
              //   <StyledTableCell component="th" scope="row">
              //     {row.name}
              //   </StyledTableCell>
              //   <StyledTableCell align="right">{row.duration}</StyledTableCell>
              //   <StyledTableCell align="right">{row.distance}</StyledTableCell>
              // </StyledTableRow>
              <AudioPlayer {...row} index={index} />
            ))
          ) : (
            <p className={classes.noData}>No Data</p>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};
export default Search;
