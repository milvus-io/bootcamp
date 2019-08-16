#! /bin/bash

grep $1 $2 | awk {'print $2'}
