#!/bin/bash

set -e

DOCFOLDER=../rustml_page/

if [ ! -e $DOCFOLDER ]; then
	echo "destination folder does not exist"
	exit 1
fi

echo "Generating doc ..."
cargo doc

rm -rf DOCFOLDER/*
cd $DOCFOLDER
git pull
cd ../rustml

echo "Syncing ..."
rsync --delete -r target/doc/* $DOCFOLDER/
rsync -r doc_data/* $DOCFOLDER/
cd $DOCFOLDER
git add -A
git commit -m "update to current version"
echo "Pushing ..."
git push

