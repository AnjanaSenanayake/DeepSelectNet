#!/bin/bash

VAR1=$1
VAR2=$2
DOWN="down"
UP="up"

if [ "$VAR2" = "$DOWN" ]; then
	rsync -auzr -P e14317@"$VAR1".ce.pdn.ac.lk:~/Workspace/Mphil /home/anjana/Documents/WorkSpace/
elif [ "$VAR2" = "$UP" ]; then
	rsync -auzpr -P /home/anjana/Documents/WorkSpace/Mphil e14317@"$VAR1".ce.pdn.ac.lk:~/Workspace/
fi