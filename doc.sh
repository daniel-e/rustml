#!/bin/bash

set -e

DOCFOLDER=../rustml_page/

if [ ! -e $DOCFOLDER ]; then
	echo "destination folder does not exist"
	exit 1
fi

echo "Generating doc ..."
cargo doc
echo "Syncing ..."
rsync --delete -r target/doc/* $DOCFOLDER/
cd $DOCFOLDER
git add *
git commit -m "update to current version"
echo "Pushing ..."
git push

