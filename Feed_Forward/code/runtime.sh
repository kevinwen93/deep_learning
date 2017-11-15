#!/bin/bash
START=$(date +%s)
# do something
# start your script work here
python2.7 demo.py 
# your logic ends here
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"